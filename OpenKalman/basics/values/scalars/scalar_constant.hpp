/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \scalar_constant.
 */

#ifndef OPENKALMAN_SCALAR_CONSTANT_HPP
#define OPENKALMAN_SCALAR_CONSTANT_HPP


namespace OpenKalman
{

#ifndef __cpp_concepts
  namespace internal
  {
    // These functions are also used in get_scalar_constant_value

    template<typename T, typename = void>
    struct has_value_member : std::false_type {};

    template<typename T>
    struct has_value_member<T, std::enable_if_t<scalar_type<decltype(T::value)>>> : std::true_type {};

    namespace detail
    {
      template<typename T, typename = void>
      struct call_result_is_scalar_impl : std::false_type {};

      template<typename T>
      struct call_result_is_scalar_impl<T, std::void_t<std::bool_constant<(T{}(), true)>>>
        : std::bool_constant<scalar_type<decltype(T{}())>> {};
    } // namespace detail

    template<typename T, typename = void>
    struct call_result_is_scalar : std::false_type {};

    template<typename T>
    struct call_result_is_scalar<T, std::enable_if_t<std::is_default_constructible_v<T>>>
      : std::bool_constant<detail::call_result_is_scalar_impl<T>::value> {};


    template<typename T, typename = void>
    struct is_runtime_scalar : std::false_type {};

    template<typename T>
    struct is_runtime_scalar<T, std::enable_if_t<scalar_type<typename std::invoke_result<T>::type>>>
      : std::true_type {};

  } // namespace internal
#endif


  namespace detail
  {
    /**
     * \brief T is a scalar constant known at compile time
     */
    template<typename T>
#ifdef __cpp_concepts
    concept compile_time_scalar_constant = std::default_initializable<std::decay_t<T>> and
      ( requires { {std::decay_t<T>::value} -> scalar_type; } or
        requires {
          {std::decay_t<T>{}()} -> scalar_type;
          typename std::bool_constant<(std::decay_t<T>{}(), true)>;
        });
#else
    constexpr bool compile_time_scalar_constant = std::is_default_constructible_v<std::decay_t<T>> and
      (internal::has_value_member<std::decay_t<T>>::value or internal::call_result_is_scalar<std::decay_t<T>>::value);
#endif


    /**
     * \brief T is a scalar constant known at runtime
     */
    template<typename T>
#ifdef __cpp_concepts
    concept runtime_scalar_constant = (not compile_time_scalar_constant<T>) and
      (scalar_type<T> or requires(std::decay_t<T> t){ {t()} -> scalar_type; });
#else
    constexpr bool runtime_scalar_constant = (not compile_time_scalar_constant<T>) and
      (scalar_type<T> or internal::is_runtime_scalar<std::decay_t<T>>::value);
#endif

  } // namespace detail


  /**
   * \brief T is a scalar constant
   * \tparam c Whether the constant is known or unknown at compile time.
   */
  template<typename T, ConstantType c = ConstantType::any>
#ifdef __cpp_concepts
  concept scalar_constant =
#else
  constexpr bool scalar_constant =
#endif
    (c == ConstantType::any and (detail::compile_time_scalar_constant<T> or detail::runtime_scalar_constant<T>)) or
    (c == ConstantType::static_constant and detail::compile_time_scalar_constant<T>) or
    (c == ConstantType::dynamic_constant and detail::runtime_scalar_constant<T>);


} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_CONSTANT_HPP
