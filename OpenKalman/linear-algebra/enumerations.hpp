/* This file is part of OpenKalman, a header-only C++ library for
* Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Enumerations relating to linear algebra.
 */

#ifndef OPENKALMAN_ENUMERATIONS_HPP
#define OPENKALMAN_ENUMERATIONS_HPP

namespace OpenKalman
{
  /**
   * \brief The type of a triangular matrix, if it is triangular.
   * \details This is generally applicable to a rank-2 tensor (e.g., a matrix).
   * It also applies to tensors of rank > 2, in which case every rank-2 slice over dimensions 0 and 1 must be a triangle of this type.
   */
  enum struct triangle_type : int {
    none, ///< Not triangular.
    lower, ///< An application should only read elements of the lower-left triangle.
    upper, ///< An application should only read elements of the upper-right triange.
    diagonal, ///< An application should only read diagonal elements. diagonal also implies both lower and upper.
    // \todo strict_diagonal,
    //< A specific diagonal object for rank 2k (k:ℕ) tensors in which each element is zero unless its
    //< indices can be divided into two identical sequences.
    //< (Examples, component x[1,0,2,1,0,2] in rank-6 tensor x or component y[2,5,2,5] in rank-4 tensor y.)
  };


  /**
   * The resulting \ref triangle_type if two \ref indexible objects, each having a triangle_type, are multiplied component-wise.
   */
  constexpr triangle_type
  operator * (triangle_type a, triangle_type b)
  {
    if (a == triangle_type::diagonal or
        b == triangle_type::diagonal or
        (a == triangle_type::upper and b == triangle_type::lower) or
        (a == triangle_type::lower and b == triangle_type::upper)) return triangle_type::diagonal;
    if (a == triangle_type::lower or
        b == triangle_type::lower) return triangle_type::lower;
    if (a == triangle_type::upper or
        b == triangle_type::upper) return triangle_type::upper;
    return triangle_type::none;
  }


  /**
   * The resulting \ref triangle_type if two \ref indexible objects, each having a triangle_type, are added component-wise.
   */
  constexpr triangle_type
  operator + (triangle_type a, triangle_type b)
  {
    if (a == triangle_type::diagonal) return b;
    if (b == triangle_type::diagonal) return a;
    if (a == triangle_type::lower and b == triangle_type::lower) return triangle_type::lower;
    if (a == triangle_type::upper and b == triangle_type::upper) return triangle_type::upper;
    return triangle_type::none;
  }


}


#endif
