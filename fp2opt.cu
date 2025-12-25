// fp2opt.cu  (NO CMAKE REQUIRED)
//
// Compile:
// nvcc fp2opt.cu -o fp2opt \
//     -std=c++17 -O3 \
//     -lOsiClp -lClp -lCoinUtils -lpthread
//
// Usage:
// ./fp2opt model.mps instance1 300
//
// Produces solutions in:
// solFiles/fp2Opt/instance1/incumbent_*.sol
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

#include "OsiClpSolverInterface.hpp"
#include "CoinPackedMatrix.hpp"
#include "CoinPackedVector.hpp"
#include "ClpSimplex.hpp"

// -------- CUDA CHECK ----------
#define CUDA_CHECK(e) { if((e)!=cudaSuccess){ \
    fprintf(stderr,"CUDA: %s\n", cudaGetErrorString(e)); exit(1); }}

// -------- DEVICE KERNELS ----------

// Feasibility check: row-wise Ax <= b
__global__ void check_feasible_csr_kernel(
    int m, int n,
    const int* __restrict__ rp,
    const int* __restrict__ ci,
    const double* __restrict__ av,
    const double* __restrict__ x,
    const double* __restrict__ rhs,
    unsigned char* __restrict__ viol
){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r>=m) return;
    int s=rp[r], e=rp[r+1];
    double sum=0;
    for(int k=s;k<e;k++) sum+=av[k]*x[ci[k]];
    viol[r] = (sum > rhs[r] + 1e-8) ? 1 : 0;
}

// 2-opt improvement kernel
struct MoveResult {
    int mutex;
    double delta;
    int i,j,di,dj;
};

__global__ void two_opt_kernel(
    int m, int n,
    const int* __restrict__ rp,
    const int* __restrict__ ci,
    const double* __restrict__ av,
    const double* __restrict__ cost,
    const double* __restrict__ rhs,
    const double* __restrict__ x,
    const double* __restrict__ activity,
    MoveResult* best
){
    int i=blockIdx.x, j=blockIdx.y;
    if(i>=n||j>=n||i>=j) return;
    if(threadIdx.x!=0) return;

    double xi=x[i], xj=x[j];
    double ci0=cost[i], cj0=cost[j];

    for(int di=-1;di<=1;di++){
        for(int dj=-1;dj<=1;dj++){
            if(!di && !dj) continue;
            double newXi=xi+di,newXj=xj+dj;
            double d= ci0*di + cj0*dj;
            if(d>=best->delta) continue;

            bool feas=true;
            for(int r=0;r<m;r++){
                int s=rp[r],e=rp[r+1];
                double ai=0,aj=0;
                for(int k=s;k<e;k++){
                    int col=ci[k];
                    if(col==i) ai=av[k];
                    if(col==j) aj=av[k];
                }
                double sum = activity[r] + ai*di + aj*dj;
                if(sum > rhs[r]+1e-8){ feas=false; break; }
            }
            if(!feas) continue;

            // lock & update
            while(atomicExch(&best->mutex,1)!=0);
            if(d<best->delta){
                best->delta=d;
                best->i=i; best->j=j;
                best->di=di; best->dj=dj;
            }
            best->mutex=0;
        }
    }
}

// -------- CPU UTILS ----------

void write_sol(const std::string& inst,int id,const std::vector<double>& x,double obj){
    std::string dir="solFiles/fp2Opt/"+inst;
    system(("mkdir -p "+dir).c_str());
    std::ofstream f(dir+"/incumbent_"+std::to_string(id)+".sol");
    f<<"obj: "<<obj<<"\n";
    for(size_t i=0;i<x.size();i++) f<<"x"<<i<<" "<<x[i]<<"\n";
    f.close();
}

std::string hash_round(const std::vector<double>& x){
    std::ostringstream o; o<<std::fixed<<std::setprecision(0);
    for(double v:x) o<<v<<",";
    return o.str();
}

bool solve_lp(OsiClpSolverInterface& s,std::vector<double>& x,double& obj){
    s.initialSolve();
    if(s.isProvenPrimalInfeasible()) return false;
    int n=s.getNumCols();
    x.assign(s.getColSolution(),s.getColSolution()+n);
    obj=s.getObjValue();
    return true;
}

void csr_from_mps(const CoinPackedMatrix* M,std::vector<int>& rp,std::vector<int>& ci,std::vector<double>& av){
    int m=M->getNumRows();
    rp.resize(m+1);
    int nz=0;
    for(int r=0;r<m;r++){
        rp[r]=nz;
        auto row=M->getVector(r);
        for(int k=0;k<row.getNumElements();k++){
            ci.push_back(row.getIndices()[k]);
            av.push_back(row.getElements()[k]);
            nz++;
        }
    }
    rp[m]=nz;
}

// -------- MAIN ----------

int main(int argc,char**argv){
    if(argc<3){ printf("usage: ./fp2opt file.mps instance [time=300]\n"); return 1; }
    std::string file=argv[1], inst=argv[2];
    int LIMIT = (argc>3?atoi(argv[3]):300);
    auto t0=std::chrono::steady_clock::now();

    OsiClpSolverInterface s;
    if(s.readMps(file.c_str())){ printf("MPS read error\n"); return 1; }

    int m=s.getNumRows(), n=s.getNumCols();
    auto A=s.getMatrixByRow();
    const double* rhs=s.getRightHandSide();
    const double* obj=s.getObjCoefficients();
    const double* lb=s.getColLower();
    const double* ub=s.getColUpper();

    std::vector<int> rp,ci;
    std::vector<double> av;
    csr_from_mps(A,rp,ci,av);

    thrust::device_vector<int> d_rp=rp, d_ci=ci;
    thrust::device_vector<double> d_av=av;
    thrust::device_vector<double> d_rhs(rhs,rhs+m);
    thrust::device_vector<double> d_obj(obj,obj+n);

    // ---- FP Start ----
    std::vector<double> xlp; double lpobj;
    if(!solve_lp(s,xlp,lpobj)){ printf("LP infeasible\n"); return 0; }

    std::unordered_set<std::string> seen;
    int it=0, inc_id=0;
    double inc_obj=1e100;
    std::vector<double> inc_x(n);

    thrust::device_vector<double> d_x(n);
    thrust::device_vector<unsigned char> d_v(m);

    while(true){
        it++;
        double T = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        if(T>LIMIT) break;

        // rounding
        std::vector<double> xr(n);
        for(int i=0;i<n;i++){
            double v=round(xlp[i]);
            if(v<lb[i]) v=lb[i];
            if(v>ub[i]) v=ub[i];
            xr[i]=v;
        }
        std::string h=hash_round(xr);
        if(seen.count(h)) break;
        seen.insert(h);

        thrust::copy(xr.begin(),xr.end(),d_x.begin());
        int BS=256,GS=(m+BS-1)/BS;
        check_feasible_csr_kernel<<<GS,BS>>>(m,n,
            thrust::raw_pointer_cast(d_rp.data()),
            thrust::raw_pointer_cast(d_ci.data()),
            thrust::raw_pointer_cast(d_av.data()),
            thrust::raw_pointer_cast(d_x.data()),
            thrust::raw_pointer_cast(d_rhs.data()),
            thrust::raw_pointer_cast(d_v.data()));
        cudaDeviceSynchronize();

        std::vector<unsigned char> hv(m);
        thrust::copy(d_v.begin(),d_v.end(),hv.begin());
        bool feas=true;
        for(auto &z:hv){ if(z){feas=false;break;} }

        double xr_obj=0; for(int i=0;i<n;i++) xr_obj+=obj[i]*xr[i];
        if(feas){
            if(xr_obj<inc_obj){
                inc_obj=xr_obj; inc_x=xr; inc_id++;
                write_sol(inst,inc_id,inc_x,inc_obj);
            }
            break;
        }

        // projection LP: simple objective replace
      ClpSimplex *model = s.getModelPtr();
for(int i = 0; i < n; i++){
    model->setObjectiveCoefficient(i, fabs(xlp[i] - xr[i]));
}

        s.initialSolve();
if(s.isProvenPrimalInfeasible()) break;
xlp.assign(s.getColSolution(), s.getColSolution() + n);
lpobj = s.getObjValue();

    }

    if(inc_id==0){ printf("no feasible integer found\n"); return 0; }

    // ---- 2-OPT ----
    thrust::device_vector<double> d_ix=inc_x;
    std::vector<double> act(m,0);
    for(int r=0;r<m;r++){
        double sum=0; for(int k=rp[r];k<rp[r+1];k++) sum+=av[k]*inc_x[ci[k]];
        act[r]=sum;
    }
    thrust::device_vector<double> d_act=act;

    MoveResult init={0,0.0,-1,-1,0,0};
    thrust::device_vector<MoveResult> d_best(1,init);

    while(true){
        MoveResult zero={0,0.0,-1,-1,0,0};
        cudaMemcpy(thrust::raw_pointer_cast(d_best.data()),&zero,sizeof(zero),cudaMemcpyHostToDevice);

        dim3 G(n,n),B(1);
        two_opt_kernel<<<G,B>>>(m,n,
            thrust::raw_pointer_cast(d_rp.data()),
            thrust::raw_pointer_cast(d_ci.data()),
            thrust::raw_pointer_cast(d_av.data()),
            thrust::raw_pointer_cast(d_obj.data()),
            thrust::raw_pointer_cast(d_rhs.data()),
            thrust::raw_pointer_cast(d_ix.data()),
            thrust::raw_pointer_cast(d_act.data()),
            thrust::raw_pointer_cast(d_best.data())
        );
        cudaDeviceSynchronize();

        MoveResult br;
        cudaMemcpy(&br, thrust::raw_pointer_cast(d_best.data()),sizeof(br),cudaMemcpyDeviceToHost);
        if(br.i<0 || br.delta>=0) break;

        inc_x[br.i]+=br.di;
        inc_x[br.j]+=br.dj;
        double newObj=0; for(int i=0;i<n;i++) newObj+=obj[i]*inc_x[i];
        if(newObj<inc_obj){
            inc_obj=newObj; inc_id++;
            write_sol(inst,inc_id,inc_x,inc_obj);
        }

        for(int r=0;r<m;r++){
            double s2=0; for(int k=rp[r];k<rp[r+1];k++) s2+=av[k]*inc_x[ci[k]];
            act[r]=s2;
        }
        thrust::copy(act.begin(),act.end(),d_act.begin());
        thrust::copy(inc_x.begin(),inc_x.end(),d_ix.begin());

        double T = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        if(T>LIMIT) break;
    }

    printf("Done. Best obj = %.10f, incumbents = %d\n",inc_obj,inc_id);
    return 0;
}
