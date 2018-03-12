/*
 Copyright (C) 2016 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file doc/COPYING.  If not see
 <http://www.gnu.org/licenses/>.
 */

#include <aspect/vof/utilities.h>

namespace aspect
{
  namespace VolumeOfFluid
  {
    using namespace dealii;

    template<>
    double vof_from_d<2> (Tensor<1, 2, double> normal,
                          double d)
    {
      const int dim = 2;
      double norm1, max, mpos, dtest;

      //Get 1-Norm
      norm1 = 0.0;
      max = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
        {
          double term = numbers::NumberTraits<double>::abs (normal[i]);
          norm1 += term;
          max = (max < term) ? term : max;
        }

      //Obtain volume
      if (d <= -0.5*norm1)
        {
          return 0.0;
        }
      if (d >= 0.5*norm1)
        {
          return 1.0;
        }
      if (norm1 == 0.0)
        {
          // Case should never occur, return reasonable output for edge circumstance
          return 0.5;
        }
      dtest = d / norm1;
      mpos = 1.0 - max/norm1;
      if (dtest < mpos - 0.5)
        {
          return (dtest + 0.5) * (dtest + 0.5) / (2.0*mpos * (1.0 - mpos));
        }
      if (dtest > 0.5 - mpos)
        {
          return 1.0 - (dtest - 0.5) * (dtest - 0.5) / (2.0*mpos * (1.0 - mpos));
        }
      return 0.5 + dtest / (1.0 - mpos);
    }

    template<>
    double d_from_vof<2> (Tensor<1, 2, double> normal,
                          double vol)
    {
      const int dim = 2;
      double norm1, max, mpos;

      //Get 1-Norm
      norm1 = 0.0;
      max = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
        {
          double term = numbers::NumberTraits<double>::abs (normal[i]);
          norm1 += term;
          max = (max < term) ? term : max;
        }

      if (norm1 == 0.0)
        {
          // For zero normal, return as if using interface parallel to an edge
          norm1 = 1.0;
          mpos = 0.0;
        }
      else
        {
          mpos = 1.0 - max/norm1;
        }

      //Obtain const
      if (vol <= 0.0)
        {
          return -0.5 * norm1;
        }
      if (vol >= 1.0)
        {
          return 0.5 * norm1;
        }
      if (vol < 0.5 * mpos / (1 - mpos))
        {
          return norm1 * (-0.5 + sqrt (2.0*vol * mpos * (1 - mpos)));
        }
      if (vol > 1.0 - 0.5 * mpos / (1 - mpos))
        {
          return norm1 * (0.5 - sqrt (2.0*(1.0 - vol) * mpos * (1 - mpos)));
        }
      return norm1 * (1 - mpos) * (vol - 0.5);

    }

    template<>
    double vof_from_d<3> (const Tensor<1, 3, double> normal,
                          const double d)
    {
      // Calculations done by Scardovelli and Zaleski in
      // doi:10.1006/jcph.2000.6567,
      // modified to fit chosen convention on normal interface
      const int dim = 3;
      double norm1, dtest;
      Tensor<1, 3, double> nnormal;
      //Simplify calculation by reducing to negative d case
      if (d>0.0)
        {
          return 1.0-vof_from_d<dim>(-normal, -d);
        }

      //Get 1-Norm
      norm1 = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
        {
          double term = numbers::NumberTraits<double>::abs (normal[i]);
          norm1 += term;
        }

      //Return volume in simple cases
      if (d <= -0.5*norm1)
        {
          return 0.0;
        }
      if (norm1 == 0.0)
        {
          // Case should never occur, return reasonable output for edge circumstance
          return 0.5;
        }
      dtest = d / norm1;

      // sort normalized values for normal into nnormal in ascending order of absolute value
      double mprod = 1.0;
      for (unsigned int i = 0; i < dim; ++i)
        {
          nnormal[i]=numbers::NumberTraits<double>::abs (normal[i])/norm1;
          mprod *= nnormal[i];
        }
      for (unsigned int i =0; i < dim; ++i)
        {
          for (unsigned int j=i+1; j < dim; ++j)
            {
              if (nnormal[j]<nnormal[i])
                {
                  //Swap
                  double tmp = nnormal[j];
                  nnormal[j] = nnormal[i];
                  nnormal[i] = tmp;
                }
            }
        }
      double m12 = nnormal[0]+nnormal[1];
      double mmin = (m12<nnormal[2])?m12:nnormal[2];
      double eps = 1e-10;
      double v1=0.0;
      if (6*nnormal[1]*nnormal[2]>eps)
        {
          v1 = nnormal[0]*nnormal[0]/(eps);
        }
      else
        {
          v1 = nnormal[0]*nnormal[0]/(6*nnormal[1]*nnormal[2]);
        }

      // do computation for standard cases
      // Case 1
      if (dtest<nnormal[0]-0.5)
        {
          return (dtest+0.5)*(dtest+0.5)*(dtest+0.5)/(6*mprod);
        }
      // Case 2
      if (dtest<nnormal[1]-0.5)
        {
          return (dtest+0.5)*(dtest+0.5-nnormal[0])/(2*nnormal[1]*nnormal[2])+v1;
        }
      // Case 3
      if (dtest<mmin-0.5)
        {
          double retVol = 0.0;
          retVol += (dtest+0.5)*(dtest+0.5)*(3*m12-dtest-0.5)/(6*mprod);
          retVol += nnormal[0]*nnormal[0]*(nnormal[0]-3*dtest-1.5)/(6*mprod);
          retVol += nnormal[1]*nnormal[1]*(nnormal[1]-3*dtest-1.5)/(6*mprod);
          return retVol;
        }
      // Case 4 m3
      if ( nnormal[2]<m12)
        {
          double retVol = 0.0;
          retVol += (dtest+0.5)*(dtest+0.5)*(3-2*dtest-1.0)/(6*mprod);
          retVol += nnormal[0]*nnormal[0]*(nnormal[0]-3*dtest-1.5)/(6*mprod);
          retVol += nnormal[1]*nnormal[1]*(nnormal[1]-3*dtest-1.5)/(6*mprod);
          retVol += nnormal[2]*nnormal[2]*(nnormal[2]-3*dtest-1.5)/(6*mprod);
          return retVol;
        }

      // Case 4 m12
      return 0.5*(2.0*dtest+1.0-m12)/nnormal[2];
    }

    template<>
    double d_from_vof<3> (const Tensor<1, 3, double> normal,
                          const double vol)
    {
      // Calculations done by Scardovelli and Zaleski in
      // doi:10.1006/jcph.2000.6567,
      // modified to fit chosen convention on normal interface
      const int dim = 3;
      double norm1;
      Tensor<1, 3, double> nnormal;
      // Simplify to vol<0.5 case
      if (vol>0.5)
        {
          return -d_from_vof<dim>(-normal, 1.0-vol);
        }

      //Get 1-Norm
      norm1 = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
        {
          double term = numbers::NumberTraits<double>::abs (normal[i]);
          norm1 += term;
        }
      double eps = 1e-10;
      double mprod = 1.0;
      if (norm1<eps)
        {
          nnormal[0] = 0.0;
          nnormal[1] = 0.0;
          nnormal[2] = 1.0;
          norm1 = 1.0;
          mprod =0.0;
        }
      else
        {
          // sort normalized values for normal into nnormal in ascending order of absolute value
          for (unsigned int i = 0; i < dim; ++i)
            {
              nnormal[i]=numbers::NumberTraits<double>::abs (normal[i])/norm1;
              mprod *= nnormal[i];
            }
          for (unsigned int i =0; i < dim; ++i)
            {
              for (unsigned int j=i+1; j < dim; ++j)
                {
                  if (nnormal[j]<nnormal[i])
                    {
                      //Swap
                      double tmp = nnormal[j];
                      nnormal[j] = nnormal[i];
                      nnormal[i] = tmp;
                    }
                }
            }
        }
      //Simple cases
      if (vol<=0.0)
        {
          return -0.5*norm1;
        }

      double m1 = nnormal[0];
      double m2 = nnormal[1];
      double m3 = nnormal[2];
      double m12 = m1+m2;

      // Case 1
      double v1=0.0;
      if (6*m2*m3>eps)
        {
          v1 = m1*m1/(eps);
        }
      else
        {
          v1 = m1*m1/(6*m2*m3);
        }
      if (vol<v1)
        {
          return -0.5+std::pow(6*mprod*vol, 1./3.);
        }

      // Case 2
      double v2 = v1 + 0.5*(m2-m1)/m3;
      if (vol<v2)
        {
          return 0.5*(-1+m1+sqrt(m1*m1+8*m2*m3*(vol-v1)));
        }

      // Case 3
      double v3 = 0.0;
      if (m12<m3)
        {
          v3 += m3*m3*(3*m12-m3)/(6.0*mprod);
          v3 += m1*m1*(m1-3*m3)/(6*mprod);
          v3 += m2*m2*(m2-3*m3)/(6*mprod);
        }
      else
        {
          v3 = 0.5*m12/m3;
        }
      if (vol<v3)
        {
          // Solve appropriate cubic
          double a2 = -3.0*m12;
          double a1 = 3.0*(m1*m1+m2*m2);
          double a0 = 6*mprod*vol-m1*m1*m1-m2*m2*m2;
          double np0 = a2*a2/9.0-a1/3.0;
          double q0 = (a1*a2-3.0*a0)/6.0-a2*a2*a2/27.0;
          double theta = acos(q0/sqrt(np0*np0*np0))/3.0;
          return sqrt(np0)*(sqrt(3.0)*sin(theta)-cos(theta))-a2/3.0;
        }

      // Case 4
      if (m3<m12)
        {
          // Solve appropriate cubic
          double a2 = -1.5;
          double a1 = 1.5*(m1*m1+m2*m2+m3*m3);
          double a0 = 6*mprod*vol-m1*m1*m1-m2*m2*m2-m3*m3*m3;
          double np0 = a2*a2/9.0-a1/3.0;
          double q0 = (a1*a2-3.0*a0)/6.0-a2*a2*a2/27.0;
          double theta = acos(q0/sqrt(np0*np0*np0))/3.0;
          return sqrt(np0)*(sqrt(3.0)*sin(theta)-cos(theta))-a2/3.0;
        }

      return -0.5+m1*vol+0.5*m12;
    }

    template<>
    void xFEM_Heaviside<2>(const int degree,
                           const Tensor<1, 2, double> normal,
                           const double d,
                           const std::vector<Point<2>> &points,
                           std::vector<double> &values)
    {
      const int basis_count=4;
      std::vector<double> coeffs(basis_count);

      const double n_xp = fabs(normal[0]), n_yp = fabs(normal[1]);
      const double sign_n_x = (((normal[0]) > 0) - ((normal[0]) < 0)),
                   sign_n_y = (((normal[1]) > 0) - ((normal[1]) < 0));

      const double norm1 = n_xp + n_yp;
      const double triangle_break = fabs(n_xp-n_yp);

      const int max_degree = 1;

      AssertThrow(degree>max_degree,
                  ExcMessage("Cannot generate xFEM polynomials are only functional for degrees<2."));

      // Values calculated using sympy
      if (d<-0.5*norm1)
        {
          for (int i =0; i < basis_count; ++i)
            coeffs[i] = 0.0;
        }
      else if (d<=-triangle_break)
        {
          //Triangle
          const double d_n = d + 0.5* (n_xp + n_yp);
          coeffs[0]=0.5L*d_n*d_n/(n_xp*n_yp); // 1
          coeffs[1]=d_n*d_n*(d_n - 1.5L*n_yp)/(n_xp*n_yp*n_yp)*sign_n_y; // 2*y - 1
          coeffs[2]=d_n*d_n*(d_n - 1.5L*n_xp)/(n_xp*n_xp*n_yp)*sign_n_x; // 2*x - 1
          coeffs[3]=1.5L*d_n*d_n*(d_n*d_n - 2*d_n*n_xp - 2*d_n*n_yp + 3*n_xp*n_yp)/(n_xp*n_xp*n_yp*n_yp)*sign_n_x*sign_n_y; // (2*x - 1)*(2*y - 1)
        }
      else if (d<triangle_break && n_xp<n_yp)
        {
          //Trapezoid X
          coeffs[0]=(d + 0.5L*n_yp)/n_yp; // 1
          coeffs[1]=0.25L*(12*d*d + n_xp*n_xp - 3*n_yp*n_yp)/(n_yp*n_yp)*sign_n_y; // 2*y - 1
          coeffs[2]=-0.5L*n_xp/n_yp*sign_n_x; // 2*x - 1
          coeffs[3]=-3*d*n_xp/(n_yp*n_yp)*sign_n_x*sign_n_y; // (2*x - 1)*(2*y - 1)
        }
      else if (d<triangle_break && n_yp<n_xp)
        {
          //Trapezoid Y
          coeffs[0]=(d + 0.5L*n_xp)/n_xp; // 1
          coeffs[1]=-0.5L*n_yp/n_xp*sign_n_y; // 2*y - 1
          coeffs[2]=0.25L*(12*pow(d, 2) - 3*n_xp*n_xp + n_yp*n_yp)/(n_xp*n_xp)*sign_n_x; // 2*x - 1
          coeffs[3]=-3*d*n_yp/(n_xp*n_xp)*sign_n_x*sign_n_y; // (2*x - 1)*(2*y - 1)
        }
      else if (d<0.5*norm1)
        {
          //ITriangle
          const double d_nn = 0.5* (n_xp + n_yp)-d;
          coeffs[0]=1.0L-0.5L*d_nn*d_nn/(n_xp*n_yp); // 1
          coeffs[1]=d_nn*d_nn*(d_nn - 1.5L*n_yp)/(n_xp*n_yp*n_yp)*sign_n_y; // 2*y - 1
          coeffs[2]=d_nn*d_nn*(d_nn - 1.5L*n_xp)/(n_xp*n_xp*n_yp)*sign_n_x; // 2*x - 1
          coeffs[3]=-1.5L*d_nn*d_nn*(d_nn*d_nn - 2*d_nn*n_xp - 2*d_nn*n_yp + 3*n_xp*n_yp)/(n_xp*n_yp*n_yp*n_yp)*sign_n_x*sign_n_y; // (2*x - 1)*(2*y - 1)
        }
      else
        {
          // Full cell
          for (int i =0; i < basis_count; ++i)
            coeffs[i] = 1.0;
        }

      for (unsigned int i = 0; i<points.size(); ++i)
        {
          const Point<2> point = points[i];
          const double x = point[0], y = point[1];
          values[i] = coeffs[0];
          if (degree>=1)
            {
              values[i] += coeffs[1]*(2*y-1.0) +
                           coeffs[2]*(2*x-1.0) +
                           coeffs[3]*(2*x - 1)*(2*y - 1);
            }
        }
    }

    template<>
    void xFEM_Heaviside_d_d<2>(const int degree,
                               const Tensor<1, 2, double> normal,
                               const double d,
                               const std::vector<Point<2>> &points,
                               std::vector<double> &values)
    {
      const int basis_count=4;
      std::vector<double> coeffs(basis_count);

      const double n_xp = fabs(normal[0]), n_yp = fabs(normal[1]);
      const double sign_n_x = (((normal[0]) > 0) - ((normal[0]) < 0)),
                   sign_n_y = (((normal[1]) > 0) - ((normal[1]) < 0));

      const double norm1 = n_xp + n_yp;
      const double triangle_break = fabs(n_xp-n_yp);

      const int max_degree = 1;

      AssertThrow(degree>max_degree,
                  ExcMessage("Cannot generate xFEM polynomials are only functional for degrees<2."));

      // Values calculated using sympy
      if (d<-0.5*norm1)
        {
          for (int i =0; i < basis_count; ++i)
            coeffs[i] = 0.0;
        }
      else if (d<=-triangle_break)
        {
          //D Triangle
          const double d_n = d + 0.5* (n_xp + n_yp);
          coeffs[0]=d_n/(n_xp*n_yp); // 1
          coeffs[1]=3*d_n*sign_n_y*(d_n - n_yp)/(n_xp*(n_yp*n_yp)); // 2*y - 1
          coeffs[2]=3*d_n*sign_n_x*(d_n - n_xp)/((n_xp*n_xp)*n_yp); // 2*x - 1
          coeffs[3]=3*d_n*sign_n_x*sign_n_y*(2*(d_n*d_n) - 3*d_n*n_xp - 3*d_n*n_yp + 3*n_xp*n_yp)/((n_xp*n_xp)*(n_yp*n_yp)); // (2*x - 1)*(2*y - 1)
        }
      else if (d<triangle_break && n_xp<n_yp)
        {
          //D Trapezoid X
          coeffs[0]=1.0/n_yp; // 1
          coeffs[1]=6*d*sign_n_y/(n_yp*n_yp); // 2*y - 1
          coeffs[2]=0; // 2*x - 1
          coeffs[3]=-3*n_xp*sign_n_x*sign_n_y/(n_yp*n_yp); // (2*x - 1)*(2*y - 1)
        }
      else if (d<triangle_break && n_yp<n_xp)
        {
          //D Trapezoid Y
          coeffs[0]=1.0/n_xp; // 1
          coeffs[1]=0; // 2*y - 1
          coeffs[2]=6*d*sign_n_x/(n_xp*n_xp); // 2*x - 1
          coeffs[3]=-3*n_yp*sign_n_x*sign_n_y/(n_xp*n_xp); // (2*x - 1)*(2*y - 1)
        }
      else if (d<0.5*norm1)
        {
          //D ITriangle
          const double d_nn = 0.5* (n_xp + n_yp)-d;
          coeffs[0]=d_nn/(n_xp*n_yp); // 1
          coeffs[1]=3*d_nn*sign_n_y*(-d_nn + n_yp)/(n_xp*(n_yp*n_yp)); // 2*y - 1
          coeffs[2]=3*d_nn*sign_n_x*(-d_nn + n_xp)/((n_xp*n_xp)*n_yp); // 2*x - 1
          coeffs[3]=3*d_nn*sign_n_x*sign_n_y*(2*(d_nn*d_nn) - 3*d_nn*n_xp - 3*d_nn*n_yp + 3*n_xp*n_yp)/((n_xp*n_xp)*(n_yp*n_yp)); // (2*x - 1)*(2*y - 1)
        }
      else
        {
          // Full cell
          for (int i =0; i < basis_count; ++i)
            coeffs[i] = 1.0;
        }

      for (unsigned int i = 0; i<points.size(); ++i)
        {
          const Point<2> point = points[i];
          const double x = point[0], y = point[1];
          values[i] = coeffs[0];
          if (degree>=1)
            {
              values[i] += coeffs[1]*(2*y-1.0) +
                           coeffs[2]*(2*x-1.0) +
                           coeffs[3]*(2*x - 1)*(2*y - 1);
            }
        }
    }


    template<int dim>
    double newton_d(const int degree,
                    const Tensor<1, dim, double> normal,
                    const double vol_frac,
                    const double epsilon,
                    const std::vector<Point<dim>> &points,
                    const std::vector<double> &weights)
    {
      double norm1=0.0;
      for (int i=0; i<dim; ++i) norm1+=fabs(normal[0]);
      double d_l=-0.5L*norm1, d_h=0.5L*norm1;
      double f_l=0.0, f_h=1.0;
      double d_guess=0.0;
      double f_guess, df_guess;

      std::vector<double> f_values(points.size());
      std::vector<double> df_values(points.size());

      for (int iter=0; iter<10; ++iter)
        {
          xFEM_Heaviside(degree, normal, d_guess, points, f_values);
          xFEM_Heavisided_d_d(degree, normal, d_guess, points, df_values);

          f_guess=0.0;
          df_guess=0.0;
          for (int i=0; i<points.size(); ++i)
            {
              f_guess  += f_values[i]*weights[i];
              df_guess += df_values[i]*weights[i];
            }

          // Break if within tolerance
          if (fabs(f_guess-vol_frac)<epsilon)
            {
              break;
            }

          if (vol_frac<f_guess)
            {
              d_h = d_guess;
              f_h = f_guess;
            }
          else
            {
              d_l = d_guess;
              f_l = f_guess;
            }

          if (fabs(df_guess)<epsilon)
            {
              d_guess = (vol_frac-f_l)/(f_h-f_l)*(d_h-d_l);
            }
          else
            {
              d_guess += (vol_frac-f_guess)/(df_guess);

              if (d_guess < d_l || d_guess > d_h)
                {
                  d_guess = (vol_frac-f_l)/(f_h-f_l)*(d_h-d_l);
                }
            }
        }

      return d_guess;
    }

    template<int dim>
    double calc_vof_flux_edge (const unsigned int dir,
                               const double timeGrad,
                               const Tensor<1, dim, double> normal,
                               const double d_face)
    {
      Tensor<1, dim, double> i_normal;
      double i_d;

      i_d = d_face+0.5*timeGrad;
      i_normal = normal;
      i_normal[dir] = timeGrad;

      return vof_from_d (i_normal, i_d);
    }
  }
}

namespace aspect
{
  namespace VolumeOfFluid
  {
#define INSTANTIATE(dim) \
  template double calc_vof_flux_edge<dim>(unsigned int dir, \
                                          double timeGrad, \
                                          Tensor<1, dim, double> normal, \
                                          double d);

    ASPECT_INSTANTIATE(INSTANTIATE)
  }
}
