#include "repair.h"
#include "activity.h"
#include "score.h"

#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include <unordered_set>

bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_fixed,
    int max_iter
)
{
    std::vector<double> activity;

    compute_constraint_activity_gpu(mip,x,activity);

    double current_score = compute_violation_score(mip, activity);

    //------------------------------------------------------------
    // PHASE 1: INTEGER REPAIR
    //------------------------------------------------------------

    for(int iter=0; iter<max_iter; iter++)
    {
        int worst_row = -1;
        double worst_violation = 0;

        for(int i=0;i<mip.num_rows;i++)
        {
            double viol = activity[i] - mip.b[i];

            if(viol > worst_violation)
            {
                worst_violation = viol;
                worst_row = i;
            }
        }

        if(worst_row == -1)
            return true;

        int start = mip.csr_row_ptr[worst_row];
        int end   = mip.csr_row_ptr[worst_row+1];

        int best_var = -1;
        double best_delta = 0;
        double best_score = current_score;

        //------------------------------------------------
        // try integer moves only
        //------------------------------------------------

        for(int k=start;k<end;k++)
        {
            int j = mip.csr_col_idx[k];

            if(is_fixed[j])
                continue;

            if(mip.vartype[j] == VarType::CONTINUOUS)
                continue;

            double old_val = x[j];

            double f = std::floor(old_val);
            double c = std::ceil(old_val);

            std::vector<double> candidates;

            if(f >= mip.lb[j]) candidates.push_back(f);
            if(c <= mip.ub[j]) candidates.push_back(c);

            for(double new_val : candidates)
            {
                double delta = new_val - old_val;
                if(delta == 0) continue;

                std::vector<double> temp_activity = activity;

                int col_start = mip.csc_col_ptr[j];
                int col_end   = mip.csc_col_ptr[j+1];

                for(int p=col_start;p<col_end;p++)
                {
                    int row = mip.csc_row_idx[p];
                    double aij = mip.csc_val[p];

                    temp_activity[row] += aij * delta;
                }

                double score = compute_violation_score(mip, temp_activity);

                if(score < best_score)
                {
                    best_score = score;
                    best_var = j;
                    best_delta = delta;
                }
            }
        }

        //------------------------------------------------
        // no integer improvement → exit phase
        //------------------------------------------------

        if(best_var == -1)
            break;

        //------------------------------------------------
        // apply move
        //------------------------------------------------

        x[best_var] += best_delta;

        int col_start = mip.csc_col_ptr[best_var];
        int col_end   = mip.csc_col_ptr[best_var+1];

        for(int p=col_start;p<col_end;p++)
        {
            int row = mip.csc_row_idx[p];
            double aij = mip.csc_val[p];

            activity[row] += aij * best_delta;
        }

        current_score = best_score;
    }

    //------------------------------------------------------------
    // PHASE 2: CONTINUOUS REPAIR
    //------------------------------------------------------------

    const double step_factor = 0.5;

    for(int iter=0; iter<max_iter; iter++)
    {
        int worst_row = -1;
        double worst_violation = 0;

        for(int i=0;i<mip.num_rows;i++)
        {
            double viol = activity[i] - mip.b[i];

            if(viol > worst_violation)
            {
                worst_violation = viol;
                worst_row = i;
            }
        }

        if(worst_row == -1)
            return true;

        int start = mip.csr_row_ptr[worst_row];
        int end   = mip.csr_row_ptr[worst_row+1];

        int best_var = -1;
        double best_delta = 0;
        double best_score = current_score;

        //------------------------------------------------
        // try continuous moves
        //------------------------------------------------

        for(int k=start;k<end;k++)
        {
            int j = mip.csr_col_idx[k];

            if(is_fixed[j])
                continue;

            if(mip.vartype[j] != VarType::CONTINUOUS)
                continue;

            double a = mip.csr_val[k];
            if(a == 0) continue;

            double old_val = x[j];

            double violation = activity[worst_row] - mip.b[worst_row];

            double delta = step_factor * (-violation / a);
            double new_val = old_val + delta;
            

            if(new_val < mip.lb[j]) new_val = mip.lb[j];
            if(new_val > mip.ub[j]) new_val = mip.ub[j];

            delta = new_val - old_val;

            if(delta == 0)
                continue;

            std::vector<double> temp_activity = activity;

            int col_start = mip.csc_col_ptr[j];
            int col_end   = mip.csc_col_ptr[j+1];

            for(int p=col_start;p<col_end;p++)
            {
                int row = mip.csc_row_idx[p];
                double aij = mip.csc_val[p];

                temp_activity[row] += aij * delta;
            }

            double score = compute_violation_score(mip, temp_activity);

            if(score < best_score)
            {
                best_score = score;
                best_var = j;
                best_delta = delta;
            }
        }

        //------------------------------------------------
        // no improvement
        //------------------------------------------------

        if(best_var == -1)
            return false;

        //------------------------------------------------
        // apply move
        //------------------------------------------------

        x[best_var] += best_delta;

        int col_start = mip.csc_col_ptr[best_var];
        int col_end   = mip.csc_col_ptr[best_var+1];

        for(int p=col_start;p<col_end;p++)
        {
            int row = mip.csc_row_idx[p];
            double aij = mip.csc_val[p];

            activity[row] += aij * best_delta;
        }

        current_score = best_score;
    }

    return false;
}

bool repair_solution_improved(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_fixed,
    int max_iter
)
{
    std::vector<double> activity;
    compute_constraint_activity_gpu(mip,x,activity);

    double current_score = compute_violation_score(mip, activity);

    const int TOP_K = 5;

    //------------------------------------------------------------
    // PHASE 1: INTEGER REPAIR
    //------------------------------------------------------------

    for(int iter=0; iter<max_iter; iter++)
    {
        std::vector<std::pair<double,int>> viol_rows;

        for(int i=0;i<mip.num_rows;i++)
        {
            double viol = activity[i] - mip.b[i];
            if(viol > 0)
                viol_rows.push_back({viol,i});
        }

        if(viol_rows.empty())
            return true;

        std::partial_sort(
            viol_rows.begin(),
            viol_rows.begin()+std::min(TOP_K,(int)viol_rows.size()),
            viol_rows.end(),
            [](auto&a,auto&b){return a.first>b.first;}
        );

        //------------------------------------------------
        // collect candidate integer variables
        //------------------------------------------------

        std::vector<int> candidate_vars;
        std::vector<char> visited(mip.num_cols,0);

        int rows_to_check = std::min(TOP_K,(int)viol_rows.size());

        for(int r=0;r<rows_to_check;r++)
        {
            int row = viol_rows[r].second;

            int start = mip.csr_row_ptr[row];
            int end   = mip.csr_row_ptr[row+1];

            for(int k=start;k<end;k++)
            {
                int j = mip.csr_col_idx[k];

                if(visited[j]) continue;
                if(is_fixed[j]) continue;
                if(mip.vartype[j]==VarType::CONTINUOUS) continue;

                visited[j]=1;
                candidate_vars.push_back(j);
            }
        }

        //------------------------------------------------
        // evaluate integer moves
        //------------------------------------------------

        int best_var=-1;
        double best_delta=0;
        double best_score=current_score;

        for(int j : candidate_vars)
        {
            double old_val = x[j];

            double f = std::floor(old_val);
            double c = std::ceil(old_val);

            double candidates[2]={f,c};

            for(double new_val : candidates)
            {
                if(new_val < mip.lb[j] || new_val > mip.ub[j])
                    continue;

                double delta=new_val-old_val;
                if(delta==0) continue;

                std::vector<double> temp_activity=activity;

                int col_start=mip.csc_col_ptr[j];
                int col_end  =mip.csc_col_ptr[j+1];

                for(int p=col_start;p<col_end;p++)
                {
                    int row=mip.csc_row_idx[p];
                    double aij=mip.csc_val[p];

                    temp_activity[row]+=aij*delta;
                }

                double score=compute_violation_score(mip,temp_activity);

                if(score < best_score)
                {
                    best_score=score;
                    best_var=j;
                    best_delta=delta;
                }
            }
        }

        if(best_var==-1)
            break;

        //------------------------------------------------
        // apply integer move
        //------------------------------------------------

        x[best_var]+=best_delta;

        int col_start=mip.csc_col_ptr[best_var];
        int col_end  =mip.csc_col_ptr[best_var+1];

        for(int p=col_start;p<col_end;p++)
        {
            int row=mip.csc_row_idx[p];
            double aij=mip.csc_val[p];

            activity[row]+=aij*best_delta;
        }

        current_score=best_score;
    }

    //------------------------------------------------------------
    // PHASE 2: CONTINUOUS REPAIR (Top-K weighted update)
    //------------------------------------------------------------

    const double step_factor=0.5;

    for(int iter=0; iter<max_iter; iter++)
    {
        std::vector<std::pair<double,int>> viol_rows;

        for(int i=0;i<mip.num_rows;i++)
        {
            double viol = activity[i]-mip.b[i];
            if(viol>0)
                viol_rows.push_back({viol,i});
        }

        if(viol_rows.empty())
            return true;

        std::partial_sort(
            viol_rows.begin(),
            viol_rows.begin()+std::min(TOP_K,(int)viol_rows.size()),
            viol_rows.end(),
            [](auto&a,auto&b){return a.first>b.first;}
        );

        //------------------------------------------------
        // collect continuous candidates
        //------------------------------------------------

        std::vector<int> candidate_vars;
        std::vector<char> visited(mip.num_cols,0);

        int rows_to_check = std::min(TOP_K,(int)viol_rows.size());

        for(int r=0;r<rows_to_check;r++)
        {
            int row = viol_rows[r].second;

            int start = mip.csr_row_ptr[row];
            int end   = mip.csr_row_ptr[row+1];

            for(int k=start;k<end;k++)
            {
                int j=mip.csr_col_idx[k];

                if(visited[j]) continue;
                if(is_fixed[j]) continue;
                if(mip.vartype[j]!=VarType::CONTINUOUS) continue;

                visited[j]=1;
                candidate_vars.push_back(j);
            }
        }

        int best_var=-1;
        double best_delta=0;
        double best_score=current_score;

        //------------------------------------------------
        // compute weighted repair step
        //------------------------------------------------

        for(int j : candidate_vars)
        {
            double old_val=x[j];

            double delta_sum=0;
            double weight_sum=0;

            for(int r=0;r<rows_to_check;r++)
            {
                int row=viol_rows[r].second;
                double viol=activity[row]-mip.b[row];

                int start=mip.csr_row_ptr[row];
                int end  =mip.csr_row_ptr[row+1];

                double aij=0;

                for(int k=start;k<end;k++)
                    if(mip.csr_col_idx[k]==j)
                        aij=mip.csr_val[k];

                if(aij==0) continue;

                delta_sum += (-viol / aij);
                weight_sum += 1.0;
            }

            if(weight_sum==0)
                continue;

            double delta = step_factor*(delta_sum/weight_sum);

            double new_val=old_val+delta;

            if(new_val<mip.lb[j]) new_val=mip.lb[j];
            if(new_val>mip.ub[j]) new_val=mip.ub[j];

            delta=new_val-old_val;

            if(delta==0) continue;

            std::vector<double> temp_activity=activity;

            int col_start=mip.csc_col_ptr[j];
            int col_end  =mip.csc_col_ptr[j+1];

            for(int p=col_start;p<col_end;p++)
            {
                int row=mip.csc_row_idx[p];
                double aij=mip.csc_val[p];

                temp_activity[row]+=aij*delta;
            }

            double score=compute_violation_score(mip,temp_activity);

            if(score < best_score)
            {
                best_score=score;
                best_var=j;
                best_delta=delta;
            }
        }

        if(best_var==-1)
            return false;

        //------------------------------------------------
        // apply move
        //------------------------------------------------

        x[best_var]+=best_delta;

        int col_start=mip.csc_col_ptr[best_var];
        int col_end  =mip.csc_col_ptr[best_var+1];

        for(int p=col_start;p<col_end;p++)
        {
            int row=mip.csc_row_idx[p];
            double aij=mip.csc_val[p];

            activity[row]+=aij*best_delta;
        }

        current_score=best_score;
    }

    return false;
}