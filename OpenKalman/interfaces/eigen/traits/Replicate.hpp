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
 * \brief Type traits as applied to Eigen::Replicate.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_REPLICATE_HPP
#define OPENKALMAN_EIGEN_TRAITS_REPLICATE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct IndexibleObjectTraits<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
  {
  private:

    using T = Eigen::Replicate<MatrixType, RowFactor, ColFactor>;

  public:

    template<Likelihood b>
    static constexpr bool is_one_by_one =
      (b != Likelihood::definitely or (RowFactor == 1 and ColFactor == 1)) and
      (RowFactor == 1 or RowFactor == Eigen::Dynamic) and
      (ColFactor == 1 or ColFactor == Eigen::Dynamic) and
      one_by_one_matrix<MatrixType, b>;

    template<Likelihood b>
    static constexpr bool is_square =
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>) and
      (RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic or
        ((RowFactor != ColFactor or square_matrix<MatrixType, b>) and
        (dynamic_dimension<MatrixType, 0> or RowFactor * index_dimension_of_v<MatrixType, 0> % ColFactor == 0) and
        (dynamic_dimension<MatrixType, 1> or ColFactor * index_dimension_of_v<MatrixType, 1> % RowFactor == 0))) and
      (has_dynamic_dimensions<MatrixType> or
        ((RowFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 0> * RowFactor % index_dimension_of_v<MatrixType, 1> == 0) and
        (ColFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 1> * ColFactor % index_dimension_of_v<MatrixType, 0> == 0)));

    static constexpr bool has_runtime_parameters = RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic;
    using type = std::tuple<typename Eigen::internal::traits<T>::MatrixTypeNested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::Replicate<equivalent_self_contained_t<MatrixType>, RowFactor, ColFactor>;
      if constexpr (not std::is_lvalue_reference_v<typename Eigen::internal::traits<N>::MatrixTypeNested>)
        return N {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (RowFactor == 1 and ColFactor == 1)
      {
        return constant_diagonal_coefficient {arg.nestedExpression()};
      }
      else if constexpr ((RowFactor == 1 or RowFactor == Eigen::Dynamic) and
        (ColFactor == 1 or ColFactor == Eigen::Dynamic) and
        constant_diagonal_matrix<MatrixType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        constant_diagonal_coefficient cd {arg.nestedExpression()};
        return internal::ScalarConstant<Likelihood::maybe, std::decay_t<decltype(cd)>> {cd};
      }
      else return std::monostate {};
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = RowFactor == 1 and ColFactor == 1 and triangular_matrix<MatrixType, t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, Likelihood::maybe>;
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_REPLICATE_HPP
