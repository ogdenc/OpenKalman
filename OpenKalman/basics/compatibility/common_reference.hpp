/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Header file for compatibility definition equivalent to std::common_reference.
 */

#ifndef OPENKALMAN_COMPATIBILITY_COMMON_REFERENCE_HPP
#define OPENKALMAN_COMPATIBILITY_COMMON_REFERENCE_HPP

#if __cplusplus < 202002L

namespace OpenKalman
{
  namespace detail
  {
    template<typename From, typename To>
    using copy_cv = std::conditional_t<
        std::is_const_v<From>,
        std::conditional_t<std::is_volatile_v<From>, const volatile To, const To>,
        std::conditional_t<std::is_volatile_v<From>, volatile To, To>
        >;

    template<typename A, typename B>
    using cond_res = decltype(false ? std::declval<A(&)()>()() : std::declval<B(&)()>()());

    template<typename A, typename B>
    using cond_res_cvref = cond_res<copy_cv<A, B>&, copy_cv<B, A>&>;


    template<typename, typename, typename = void>
    struct common_ref {};

    template<typename A, typename B>
    using common_ref_t = typename common_ref<A, B>::type;

    template<typename A, typename B>
    struct common_ref<A&, B&, std::void_t<cond_res_cvref<A, B>>> : std::enable_if<
      std::is_reference_v<cond_res_cvref<A, B>>, cond_res_cvref<A, B>> {};

    template<typename A, typename B>
    using common_ref_C = std::remove_reference_t<common_ref_t<A&, B&>>&&;

    template<typename A, typename B>
    struct common_ref<A&&, B&&, std::enable_if_t<std::is_convertible_v<A&&, common_ref_C<A, B>> and std::is_convertible_v<B&&, common_ref_C<A, B>>>>
    { using type = common_ref_C<A, B>; };

    template<typename A, typename B>
    using common_ref_D = common_ref_t<const A&, B&>;

    template<typename A, typename B>
    struct common_ref<A&&, B&, std::enable_if_t<std::is_convertible_v<A&&, common_ref_D<A, B>>>>
    { using type = common_ref_D<A, B>; };

    template<typename A, typename B>
    struct common_ref<A&, B&&> : common_ref<B&&, A&> {};

    
    template<typename T1, typename T2, int b = 1, typename = void>
    struct common_reference_impl : common_reference_impl<T1, T2, b + 1> {};
    
    template<typename T1, typename T2>
    struct common_reference_impl<T1&, T2&, 1, std::void_t<common_ref_t<T1&, T2&>>> { using type = common_ref_t<T1&, T2&>; };

    template<typename T1, typename T2>
    struct common_reference_impl<T1&&, T2&&, 1, std::void_t<common_ref_t<T1&&, T2&&>>> { using type = common_ref_t<T1&&, T2&&>; };

    template<typename T1, typename T2>
    struct common_reference_impl<T1&, T2&&, 1, std::void_t<common_ref_t<T1&, T2&&>>> { using type = common_ref_t<T1&, T2&&>; };

    template<typename T1, typename T2>
    struct common_reference_impl<T1&&, T2&, 1, std::void_t<common_ref_t<T1&&, T2&>>> { using type = common_ref_t<T1&&, T2&>; };

    template<typename T1, typename T2>
    struct common_reference_impl<T1, T2, 3, std::void_t<cond_res<T1, T2>>> { using type = cond_res<T1, T2>; };

    template<typename T1, typename T2>
    struct common_reference_impl<T1, T2, 4, std::void_t<std::common_type_t<T1, T2>>> { using type = std::common_type_t<T1, T2>; };

    template<typename T1, typename T2>
    struct common_reference_impl<T1, T2, 5, void> {};

  }


  template<typename...T>
  struct common_reference;

  template<>
  struct common_reference<> {};

  template<typename T>
  struct common_reference<T> { using type = T; };

  template<typename T1, typename T2, typename...Ts>
  struct common_reference<T1, T2, Ts...> : common_reference<detail::common_reference_impl<T1, T2>, Ts...> {};


  template<typename...T>
  using common_reference_t = typename common_reference<T...>::type;

}

#endif

#endif //OPENKALMAN_COMPATIBILITY_COMMON_REFERENCE_HPP
