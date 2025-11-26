/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to native Eigen::TriangularView.
 */

#ifndef OPENKALMAN_EIGEN_TRIANGULARVIEW_HPP
#define OPENKALMAN_EIGEN_TRIANGULARVIEW_HPP

#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/traits/triangle_type_of.hpp"

namespace OpenKalman
{
  namespace interface
  {
    template<typename MatrixType, unsigned int Mode>
    struct object_traits<Eigen::TriangularView<MatrixType, Mode>>
    {
    private:

      using Xpr = Eigen::TriangularView<MatrixType, Mode>;
      using IndexType = typename MatrixType::Index;

    public:

      using scalar_type = scalar_type_of_t<MatrixType>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(arg.nestedExpression()); }


      template<typename Arg, typename N>
      static constexpr auto get_pattern_collection(const Arg& arg, N n)
      {
        return OpenKalman::get_pattern_collection(arg.nestedExpression(), n);
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nestedExpression();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (zero<MatrixType> or ((Mode & Eigen::ZeroDiag) != 0 and diagonal_matrix<MatrixType>))
          return values::fixed_value<scalar_type_of_t<MatrixType>, 0>{};
        else
          return std::monostate{};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        using Scalar = scalar_type_of_t<MatrixType>;

        if constexpr ((Mode & Eigen::UnitDiag) != 0 and (
          ((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, triangle_type::lower>) or
          ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, triangle_type::upper>)))
        {
          return values::fixed_value<Scalar, 1>{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) != 0 and (
          ((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, triangle_type::lower>) or
          ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, triangle_type::upper>)))
        {
          return values::fixed_value<Scalar, 0>{};
        }
        else
        {
          return constant_diagonal_value {arg.nestedExpression()};
        }
      }


      template<applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;


      template<applicability b>
      static constexpr bool is_square = square_shaped<MatrixType, b>;

    private:

      static constexpr triangle_type Eigen_tri =
        ((Mode & Eigen::Lower) != 0) ? triangle_type::lower :
        ((Mode & Eigen::Upper) != 0) ? triangle_type::upper :
        triangle_type::none;

    public:

      static constexpr triangle_type triangle_type_value = Eigen_tri * triangle_type_of_v<MatrixType>;


      static constexpr bool is_triangular_adapter = true;


      static constexpr bool is_hermitian = diagonal_matrix<MatrixType> and (not values::complex<scalar_type> or
        values::not_complex<constant_value<MatrixType>> or values::not_complex<constant_diagonal_value<MatrixType>>);

    };

  }

}

#endif
