/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::DiagonalMatrix.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN_TRAITS_DIAGONALMATRIX_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace interface
  {
    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct indexible_object_traits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
      : Eigen3::indexible_object_traits_base<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
    private:

      using Xpr = Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>;
      using Base = Eigen3::indexible_object_traits_base<Xpr>;

    public:

      template<typename Arg>
      static constexpr auto
      count_indices(const Arg& arg)
      {
        if constexpr (SizeAtCompileTime == 1)
          return std::integral_constant<std::size_t, 0_uz>{};
        else
          return std::integral_constant<std::size_t, 2_uz>{};
      }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N)
      {
        if constexpr (SizeAtCompileTime == Eigen::Dynamic) return static_cast<std::size_t>(arg.rows());
        else return Dimensions<SizeAtCompileTime>{};
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        // Note, early Eigen versions, at least, do not have rvalue-qualified "diagonal()" member functions.
        if constexpr (std::is_rvalue_reference_v<Arg&&>)
          return std::move(arg.diagonal());
        else
          return arg.diagonal();
      }


      // get_constant() not defined


      // get_constant_diagonal() not defined


      template<Applicability b>
      static constexpr bool one_dimensional = SizeAtCompileTime == 1 or (SizeAtCompileTime == Eigen::Dynamic and b == Applicability::permitted);


      template<Applicability b>
      static constexpr bool is_square = true;


      template<TriangleType t>
      static constexpr bool is_triangular = true;


      static constexpr bool is_triangular_adapter = false;


      // is_hermitian not defined because matrix is diagonal;


      // make_hermitian_adapter(Arg&& arg) not defined

    };

  } // namespace interface

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN_TRAITS_DIAGONALMATRIX_HPP
