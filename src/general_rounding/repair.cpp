#include "repair.h"
#include "activity.h"

#include <cmath>
#include <limits>
#include <iostream>

// bool repair_solution(
//     const MIPProblem& mip,
//     std::vector<double>& x,
//     int max_iter
// )
// {
//     std::vector<double> activity;

//     compute_constraint_activity_gpu(mip,x,activity);

//     double current_score = 0.0;

//     for(int i=0;i<mip.num_rows;i++)
//     {
//         double viol = activity[i] - mip.b[i];
//         if(viol > 0) current_score += viol*viol;
//     }

//     for(int iter=0; iter < max_iter; iter++)
//     {
//         //------------------------------------------------
//         // Find most violated constraint
//         //------------------------------------------------

//         int worst_row = -1;
//         double worst_violation = 0.0;

//         for(int i=0;i<mip.num_rows;i++)
//         {
//             double viol = activity[i] - mip.b[i];

//             if(viol > worst_violation)
//             {
//                 worst_violation = viol;
//                 worst_row = i;
//             }
//         }

//         if(worst_row == -1)
//             return true;

//         //------------------------------------------------
//         // Search best improving variable
//         //------------------------------------------------

//         int start = mip.csr_row_ptr[worst_row];
//         int end   = mip.csr_row_ptr[worst_row+1];

//         int best_var = -1;
//         double best_delta = 0;
//         double best_score = current_score;

//         for(int k=start;k<end;k++)
//         {
//             int j = mip.csr_col_idx[k];
//             double a = mip.csr_val[k];

//             double old_val = x[j];

//             std::vector<double> candidates;

//             //------------------------------------------------
//             // try floor
//             //------------------------------------------------

//             double f = std::floor(old_val);
//             if(f >= mip.lb[j])
//                 candidates.push_back(f);

//             //------------------------------------------------
//             // try ceil
//             //------------------------------------------------

//             double c = std::ceil(old_val);
//             if(c <= mip.ub[j])
//                 candidates.push_back(c);

//             //------------------------------------------------

//             for(double new_val : candidates)
//             {
//                 double delta = new_val - old_val;

//                 if(delta == 0)
//                     continue;

//                 //------------------------------------------------
//                 // simulate activity update
//                 //------------------------------------------------

//                 std::vector<double> temp_activity = activity;

//                 int col_start = mip.csc_col_ptr[j];
//                 int col_end   = mip.csc_col_ptr[j+1];

//                 for(int p=col_start;p<col_end;p++)
//                 {
//                     int row = mip.csc_row_idx[p];
//                     double aij = mip.csc_val[p];

//                     temp_activity[row] += aij * delta;
//                 }

//                 //------------------------------------------------
//                 // compute score
//                 //------------------------------------------------

//                 double score = 0.0;

//                 for(int r=0;r<mip.num_rows;r++)
//                 {
//                     double viol = temp_activity[r] - mip.b[r];
//                     if(viol > 0)
//                         score += viol*viol;
//                 }

//                 //------------------------------------------------
//                 // improvement
//                 //------------------------------------------------

//                 if(score < best_score)
//                 {
//                     best_score = score;
//                     best_var = j;
//                     best_delta = delta;
//                 }
//             }
//         }

//         //------------------------------------------------
//         // no improving move
//         //------------------------------------------------

//         if(best_var == -1)
//             return false;

//         //------------------------------------------------
//         // apply best move
//         //------------------------------------------------

//         double old_val = x[best_var];
//         double new_val = old_val + best_delta;

//         x[best_var] = new_val;

//         //------------------------------------------------
//         // update activity incrementally
//         //------------------------------------------------

//         int col_start = mip.csc_col_ptr[best_var];
//         int col_end   = mip.csc_col_ptr[best_var+1];

//         for(int p=col_start;p<col_end;p++)
//         {
//             int row = mip.csc_row_idx[p];
//             double aij = mip.csc_val[p];

//             activity[row] += aij * best_delta;
//         }

//         current_score = best_score;
//     }

//     return false;
// }





// bool repair_solution(
//     const MIPProblem& mip,
//     std::vector<double>& x,
//     const std::vector<bool>& is_fixed,
//     int max_iter
// )
// {
//     std::vector<double> activity;

//     compute_constraint_activity_gpu(mip,x,activity);

//     auto compute_score = [&](const std::vector<double>& act)
//     {
//         double score = 0.0;

//         for(int i=0;i<mip.num_rows;i++)
//         {
//             double viol = act[i] - mip.b[i];
//             if(viol > 0)
//                 score += viol*viol;
//         }

//         return score;
//     };

//     double current_score = compute_score(activity);

//     for(int iter=0; iter<max_iter; iter++)
//     {
//         //------------------------------------------------
//         // find most violated constraint
//         //------------------------------------------------

//         int worst_row = -1;
//         double worst_violation = 0;

//         for(int i=0;i<mip.num_rows;i++)
//         {
//             double viol = activity[i] - mip.b[i];

//             if(viol > worst_violation)
//             {
//                 worst_violation = viol;
//                 worst_row = i;
//             }
//         }

//         if(worst_row == -1)
//             return true;

//         //------------------------------------------------
//         // search best improving move
//         //------------------------------------------------

//         int start = mip.csr_row_ptr[worst_row];
//         int end   = mip.csr_row_ptr[worst_row+1];

//         int best_var = -1;
//         double best_delta = 0;
//         double best_score = current_score;

//         for(int k=start;k<end;k++)
//         {
//             int j = mip.csr_col_idx[k];
//             double a = mip.csr_val[k];

//             if(is_fixed[j])
//                 continue;

//             double old_val = x[j];

//             //------------------------------------------------
//             // integer variables
//             //------------------------------------------------

//             if(mip.vartype[j] != VarType::CONTINUOUS)
//             {
//                 std::vector<double> candidates;

//                 double f = std::floor(old_val);
//                 double c = std::ceil(old_val);

//                 if(f >= mip.lb[j]) candidates.push_back(f);
//                 if(c <= mip.ub[j]) candidates.push_back(c);

//                 for(double new_val : candidates)
//                 {
//                     double delta = new_val - old_val;
//                     if(delta == 0) continue;

//                     std::vector<double> temp_activity = activity;

//                     int col_start = mip.csc_col_ptr[j];
//                     int col_end   = mip.csc_col_ptr[j+1];

//                     for(int p=col_start;p<col_end;p++)
//                     {
//                         int row = mip.csc_row_idx[p];
//                         double aij = mip.csc_val[p];

//                         temp_activity[row] += aij * delta;
//                     }

//                     double score = compute_score(temp_activity);

//                     if(score < best_score)
//                     {
//                         best_score = score;
//                         best_var = j;
//                         best_delta = delta;
//                     }
//                 }
//             }

//             //------------------------------------------------
//             // continuous variables
//             //------------------------------------------------

//             else
//             {
//                 if(a == 0) continue;

//                 double violation = activity[worst_row] - mip.b[worst_row];

//                 double delta = -violation / a;

//                 double new_val = old_val + delta;

//                 if(new_val < mip.lb[j]) new_val = mip.lb[j];
//                 if(new_val > mip.ub[j]) new_val = mip.ub[j];

//                 delta = new_val - old_val;

//                 if(delta == 0) continue;

//                 std::vector<double> temp_activity = activity;

//                 int col_start = mip.csc_col_ptr[j];
//                 int col_end   = mip.csc_col_ptr[j+1];

//                 for(int p=col_start;p<col_end;p++)
//                 {
//                     int row = mip.csc_row_idx[p];
//                     double aij = mip.csc_val[p];

//                     temp_activity[row] += aij * delta;
//                 }

//                 double score = compute_score(temp_activity);

//                 if(score < best_score)
//                 {
//                     best_score = score;
//                     best_var = j;
//                     best_delta = delta;
//                 }
//             }
//         }

//         //------------------------------------------------
//         // no improving move
//         //------------------------------------------------

//         if(best_var == -1)
//             return false;

//         //------------------------------------------------
//         // apply move
//         //------------------------------------------------

//         x[best_var] += best_delta;

//         int col_start = mip.csc_col_ptr[best_var];
//         int col_end   = mip.csc_col_ptr[best_var+1];

//         for(int p=col_start;p<col_end;p++)
//         {
//             int row = mip.csc_row_idx[p];
//             double aij = mip.csc_val[p];

//             activity[row] += aij * best_delta;
//         }

//         current_score = best_score;
//     }
//     return false;
// }




bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_fixed,
    int max_iter
)
{
    std::vector<double> activity;

    compute_constraint_activity_gpu(mip,x,activity);

    auto compute_score = [&](const std::vector<double>& act)
    {
        double score = 0.0;

        for(int i=0;i<mip.num_rows;i++)
        {
            double viol = act[i] - mip.b[i];
            if(viol > 0)
                // score += viol*viol;
                score += viol;
        }

        return score;
    };

    double current_score = compute_score(activity);

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

                double score = compute_score(temp_activity);

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

            double score = compute_score(temp_activity);

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