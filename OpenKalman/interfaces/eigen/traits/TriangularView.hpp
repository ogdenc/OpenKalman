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

#include <type_traits>


namespace OpenKalman
{
  namespace interface
  {
    template<typename MatrixType, unsigned int Mode>
    struct indexible_object_traits<Eigen::TriangularView<MatrixType, Mode>>
    {
    private:

      using Xpr = Eigen::TriangularView<MatrixType, Mode>;
      using IndexType = typename MatrixType::Index;

    public:

      using scalar_type = scalar_type_of_t<MatrixType>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(arg.nestedExpression()); }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
      {
        return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
      }


      using dependents = std::tuple<typename Eigen::internal::traits<Xpr>::MatrixTypeNested>;


      static constexpr bool has_runtime_parameters = false;


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nestedExpression();
      }


      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return TriangularMatrix<equivalent_self_contained_t<MatrixType>, triangle_type_of_v<Arg>> {std::forward<Arg>(arg)};
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (zero<MatrixType> or ((Mode & Eigen::ZeroDiag) != 0 and diagonal_matrix<MatrixType, Likelihood::maybe>))
          return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
        else
          return std::monostate{};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        constexpr auto b = has_dynamic_dimensions<MatrixType> ? Likelihood::maybe : Likelihood::definitely;

        if constexpr (not square_shaped<MatrixType, Likelihood::maybe>)
        {
          return std::monostate{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) == 0 and Eigen3::eigen_Identity<MatrixType>)
        {
          return internal::ScalarConstant<b, Scalar, 1>{};
        }
        else if constexpr (((Mode & Eigen::UnitDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, TriangleType::lower, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, TriangleType::upper, Likelihood::maybe>))))
        {
          return internal::ScalarConstant<b, Scalar, 1>{};
        }
        else if constexpr ((Mode & Eigen::ZeroDiag) != 0 and
          (((Mode & Eigen::Upper) != 0 and triangular_matrix<MatrixType, TriangleType::lower, Likelihood::maybe>) or
            ((Mode & Eigen::Lower) != 0 and triangular_matrix<MatrixType, TriangleType::upper, Likelihood::maybe>)))
        {
          return internal::ScalarConstant<b, Scalar, 0>{};
        }
        else
        {
          return constant_diagonal_coefficient {arg.nestedExpression()};
        }
      }


      template<Likelihood b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;


      template<Likelihood b>
      static constexpr bool is_square = square_shaped<MatrixType, b>;


      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular =
        (t == TriangleType::lower and ((Mode & Eigen::Lower) != 0 or triangular_matrix<MatrixType, TriangleType::lower, b>)) or
        (t == TriangleType::upper and ((Mode & Eigen::Upper) != 0 or triangular_matrix<MatrixType, TriangleType::upper, b>)) or
        (t == TriangleType::diagonal and triangular_matrix<MatrixType, (Mode & Eigen::Lower) ? TriangleType::upper : TriangleType::lower, b>) or
        (t == TriangleType::any and square_shaped<MatrixType, b>);


      static constexpr bool is_triangular_adapter = true;


      static constexpr bool is_hermitian = diagonal_matrix<MatrixType> and
        (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar> or
          real_axis_number<constant_coefficient<MatrixType>> or
          real_axis_number<constant_diagonal_coefficient<MatrixType>>);

    };

  } // namespace interface

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN_TRAITS_HPP
