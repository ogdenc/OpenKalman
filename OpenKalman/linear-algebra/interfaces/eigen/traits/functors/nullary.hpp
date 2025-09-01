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
 * \brief Definitions for \ref OpenKalman::Eigen3::NullaryFunctorTraits.
 */

#ifndef OPENKALMAN_EIGEN_NULLARYFUNCTORTRAITS_HPP
#define OPENKALMAN_EIGEN_NULLARYFUNCTORTRAITS_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  // Default nullary traits, if NullaryOp is not specifically matched.
  template<typename NullaryOp, typename PlainObjectType>
  struct NullaryFunctorTraits
  {
    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg) { return arg.functor()(); }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg&) { return std::monostate {}; }

    template<triangle_type t>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar, typename PlainObjectType>
  struct NullaryFunctorTraits<Eigen::internal::scalar_identity_op<Scalar>, PlainObjectType>
  {
    template<typename Arg>
    static constexpr auto get_constant(const Arg&) { return std::monostate {}; }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg&) { return values::fixed_value<Scalar, 1>{}; }

    template<triangle_type t>
    static constexpr bool is_triangular = true;

    static constexpr bool is_hermitian = true;
  };


  template<typename Scalar, typename PlainObjectType>
  struct NullaryFunctorTraits<Eigen::internal::linspaced_op<Scalar>, PlainObjectType>
  {
    template<typename Arg>
    static constexpr auto get_constant(const Arg&) { return std::monostate {}; }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg&) { return std::monostate {}; }

    template<triangle_type t>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar, typename PlainObjectType>
  struct NullaryFunctorTraits<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>
  {
    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg) { return arg.functor()(); }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg&) { return std::monostate {}; }

    template<triangle_type t>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = not values::complex<Scalar>;
  };

}

#endif
