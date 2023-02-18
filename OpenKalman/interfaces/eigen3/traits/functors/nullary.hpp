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

#ifndef OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_NULLARY_HPP
#define OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_NULLARY_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  namespace EGI = Eigen::internal;


  template<typename Scalar, typename PlainObjectType>
  struct FunctorTraits<EGI::scalar_identity_op<Scalar>, PlainObjectType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (std::is_same_v<T<Arg>, constant_diagonal_coefficient<Arg>>)
        return std::integral_constant<int, 1>{};
      else
        return std::monostate {};
    }

    static constexpr bool is_diagonal = square_matrix<PlainObjectType>;

    static constexpr TriangleType triangle_type = square_matrix<PlainObjectType> ? TriangleType::diagonal : TriangleType::none;

    static constexpr bool is_hermitian = square_matrix<PlainObjectType>;
  };


  template<typename Scalar, typename PlainObjectType>
  struct FunctorTraits<EGI::linspaced_op<Scalar>, PlainObjectType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg) { return std::monostate {}; }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar, typename PlainObjectType>
  struct FunctorTraits<EGI::scalar_constant_op<Scalar>, PlainObjectType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (std::is_same_v<T<Arg>, constant_coefficient<Arg>>)
        return arg.functor()();
      else
        return std::monostate {};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = square_matrix<PlainObjectType> and not complex_number<Scalar>;
  };

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_NULLARY_HPP
