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
    struct IndexibleObjectTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<MatrixType>;

      template<std::size_t N, typename Arg>
      static constexpr auto get_index_descriptor(const Arg& arg)
      {
        return OpenKalman::get_index_descriptor<N>(arg.nestedExpression());
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<MatrixType, b>;

      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename Eigen::internal::traits<Eigen::TriangularView<MatrixType, Mode>>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
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
        if constexpr (zero_matrix<MatrixType> or ((Mode & Eigen::ZeroDiag) != 0 and diagonal_matrix<MatrixType, Likelihood::maybe>))
          return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<MatrixType>, 0>{};
        else
          return std::monostate{};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        using Scalar = scalar_type_of_t<MatrixType>;
        constexpr auto b = has_dynamic_dimensions<MatrixType> ? Likelihood::maybe : Likelihood::definitely;

        if constexpr (not square_matrix<MatrixType, Likelihood::maybe>)
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

      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular =
        (t == TriangleType::lower and ((Mode & Eigen::Lower) != 0 or triangular_matrix<MatrixType, TriangleType::lower, b>)) or
        (t == TriangleType::upper and ((Mode & Eigen::Upper) != 0 or triangular_matrix<MatrixType, TriangleType::upper, b>)) or
        (t == TriangleType::diagonal and triangular_matrix<MatrixType, (Mode & Eigen::Lower) ? TriangleType::upper : TriangleType::lower, b>) or
        (t == TriangleType::any and square_matrix<MatrixType, b>);

      static constexpr bool is_triangular_adapter = true;

      // make_triangular_matrix not included because TriangularView is already triangular if square.

      static constexpr bool is_hermitian = diagonal_matrix<MatrixType> and
        (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar> or
          real_axis_number<constant_coefficient<MatrixType>> or
          real_axis_number<constant_diagonal_coefficient<MatrixType>>);

      template<HermitianAdapterType t, typename Arg>
      static constexpr auto make_hermitian_adapter(Arg&& arg)
      {
        constexpr auto HMode = t == HermitianAdapterType::upper ? Eigen::Upper : Eigen::Lower;
        constexpr auto TriMode = (Mode & Eigen::Upper) != 0 ? Eigen::Upper : Eigen::Lower;
        if constexpr (HMode == TriMode)
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().template selfadjointView<HMode>());
        else
          return make_self_contained<Arg>(std::forward<Arg>(arg).nestedExpression().adjoint().template selfadjointView<HMode>());
      }


      using scalar_type = scalar_type_of_t<MatrixType>;


      template<typename Arg>
      static scalar_type_of_t<Arg> get(const Arg& arg, Eigen::Index i, Eigen::Index j)
      {
        if ((i > j and (Mode & Eigen::Upper) != 0) or (i < j and (Mode & Eigen::Lower) != 0)) return 0;
        else return arg.coeff(i, j);
      }


  #ifdef __cpp_concepts
      template<typename Arg> requires ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0)
  #else
      template<typename Arg, std::enable_if_t<((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0), int> = 0>
  #endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, Eigen::Index i, Eigen::Index j)
      {
        arg.coeffRef(i, j) = s;
      }
    };

  } // namespace interface

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN_TRAITS_HPP
