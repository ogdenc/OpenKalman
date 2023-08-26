/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Trait details for Eigen nullary functors.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_FUNCTORS_NULLARY_HPP
#define OPENKALMAN_EIGEN_TRAITS_FUNCTORS_NULLARY_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  template<typename Scalar, typename PlainObjectType>
  struct FunctorTraits<Eigen::internal::scalar_identity_op<Scalar>, PlainObjectType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      constexpr auto b = has_dynamic_dimensions<Arg> ? Likelihood::maybe : Likelihood::definitely;
      if constexpr (is_diag) return internal::ScalarConstant<b, Scalar, 1>{};
      else return std::monostate {};
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = square_matrix<PlainObjectType, b>;

    static constexpr bool is_hermitian = square_matrix<PlainObjectType>;
  };


  template<typename Scalar, typename PlainObjectType>
  struct FunctorTraits<Eigen::internal::linspaced_op<Scalar>, PlainObjectType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg) { return std::monostate {}; }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar, typename PlainObjectType>
  struct FunctorTraits<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return arg.functor()();
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = square_matrix<PlainObjectType> and not complex_number<Scalar>;
  };

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN_TRAITS_FUNCTORS_NULLARY_HPP
