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

#ifndef OPENKALMAN_COMPATIBILITY_COMMON_HPP
#define OPENKALMAN_COMPATIBILITY_COMMON_HPP

#include "core-concepts.hpp"

namespace OpenKalman::stdcompat
{
#if __cplusplus >= 202002L
  using std::common_reference;
  using std::common_reference_t;
  using std::common_reference_with;
  using std::common_with;
#else
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
    struct common_ref<A&&, B&&, std::enable_if_t<stdcompat::convertible_to<A&&, common_ref_C<A, B>> and stdcompat::convertible_to<B&&, common_ref_C<A, B>>>>
    { using type = common_ref_C<A, B>; };

    template<typename A, typename B>
    using common_ref_D = common_ref_t<const A&, B&>;

    template<typename A, typename B>
    struct common_ref<A&&, B&, std::enable_if_t<stdcompat::convertible_to<A&&, common_ref_D<A, B>>>>
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

  namespace detail
  {
    template<typename T1, typename T2, typename = void, typename...Ts>
    struct combine_common_reference {};

    template<typename T1, typename T2, typename...Ts>
    struct combine_common_reference<T1, T2, std::void_t<typename common_reference_impl<T1, T2>::type>, Ts...>
      : common_reference<typename common_reference_impl<T1, T2>::type, Ts...> {};
  }

  template<typename T1, typename T2, typename...Ts>
  struct common_reference<T1, T2, Ts...> : detail::combine_common_reference<T1, T2, void, Ts...> {};


  template<typename...T>
  using common_reference_t = typename common_reference<T...>::type;


  template<typename T, typename U>
  inline constexpr bool
  common_reference_with =
    same_as<common_reference_t<T, U>, common_reference_t<U, T>> and
    stdcompat::convertible_to<T, common_reference_t<T, U>> and
    stdcompat::convertible_to<U, common_reference_t<T, U>>;

  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct common_with_impl1 : std::false_type {};

    template<typename T, typename U>
    struct common_with_impl1<T, U, std::enable_if_t<
      (same_as<typename std::common_type<T, U>::type, typename std::common_type<U, T>::type>)>> : std::true_type {};

    template<typename T, typename U, typename = void>
    struct common_with_impl2 : std::false_type {};

    template<typename T, typename U>
    struct common_with_impl2<T, U, std::void_t<
      decltype(static_cast<typename std::common_type<T, U>::type>(std::declval<T>())),
      decltype(static_cast<typename std::common_type<T, U>::type>(std::declval<U>()))>> : std::true_type {};

    template<typename T, typename U, typename = void>
    struct common_with_impl3 : std::false_type {};

    template<typename T, typename U>
    struct common_with_impl3<T, U, std::enable_if_t<
      (common_reference_with<std::add_lvalue_reference_t<typename std::common_type<T, U>::type>,
      common_reference_t<std::add_lvalue_reference_t<const T>, std::add_lvalue_reference_t<const U>>>)>> : std::true_type {};
  }


  template<typename T, typename U>
  inline constexpr bool
  common_with =
    detail::common_with_impl1<T, U>::value and
    detail::common_with_impl2<T, U>::value and
    stdcompat::common_reference_with<std::add_lvalue_reference_t<const T>, std::add_lvalue_reference_t<const U>> and
    detail::common_with_impl3<T, U>::value;

#endif
}


namespace OpenKalman::stdcompat
{
#ifdef __cpp_lib_concepts
  using std::assignable_from;
#else
  namespace detail
  {
    template<typename LHS, typename RHS, typename = void>
    struct assignable_from_impl : std::false_type {};

    template<typename LHS, typename RHS>
    struct assignable_from_impl<LHS, RHS, std::enable_if_t<
      stdcompat::same_as<decltype(std::declval<LHS>() = std::declval<RHS&&>()), LHS>>> : std::true_type {};
  }


  template<typename LHS, typename RHS>
  inline constexpr bool
  assignable_from =
    std::is_lvalue_reference_v<LHS> and
    stdcompat::common_reference_with<const std::remove_reference_t<LHS>&, const std::remove_reference_t<RHS>&> and
    detail::assignable_from_impl<LHS, RHS>::value;
#endif
}


namespace OpenKalman::stdcompat::ranges
{
#if __cplusplus >= 202002L
  using std::ranges::swap;
#else
  namespace detail_swap
  {
    template<typename Tp>
    inline constexpr bool class_or_enum = std::is_class_v<Tp> or std::is_union_v<Tp> or std::is_enum_v<Tp>;


    template<typename Tp> void swap(Tp&, Tp&) noexcept = delete;


    template<typename Tp, typename Up, typename = void>
    struct adl_swap_impl : std::false_type {};

    template<typename Tp, typename Up>
    struct adl_swap_impl<Tp, Up, std::void_t<
      decltype(swap(static_cast<Tp&&>(std::declval<Tp&>()), static_cast<Up&&>(std::declval<Up&>())))>> : std::true_type {};


    template<typename Tp, typename Up>
    inline constexpr bool adl_swap =
      (class_or_enum<std::remove_reference_t<Tp>> or class_or_enum<std::remove_reference_t<Up>>) and
      adl_swap_impl<Tp, Up>::value;


    class swap_impl
    {
      template<typename Tp, typename Up>
      static constexpr bool S_noexcept()
      {
        if constexpr (adl_swap<Tp, Up>)
          return noexcept(swap(std::declval<Tp>(), std::declval<Up>()));
        else
          return std::is_nothrow_move_constructible_v<std::remove_reference_t<Tp>> and
            std::is_nothrow_move_assignable_v<std::remove_reference_t<Tp>>;
      }

    public:

      template<typename Tp, typename Up, std::enable_if_t<adl_swap<Tp, Up> or
        (stdcompat::same_as<Tp, Up> && std::is_lvalue_reference_v<Tp> and
            stdcompat::move_constructible<std::remove_reference_t<Tp>> and
            stdcompat::assignable_from<Tp, std::remove_reference_t<Tp>>), int> = 0>
      constexpr void
      operator()(Tp&& t, Up&& u) const noexcept(S_noexcept<Tp, Up>())
      {
        if constexpr (adl_swap<Tp, Up>)
        {
          swap(static_cast<Tp&&>(t), static_cast<Up&&>(u));
        }
        else
        {
          auto tmp = static_cast<std::remove_reference_t<Tp>&&>(t);
          t = static_cast<std::remove_reference_t<Tp>&&>(u);
          u = static_cast<std::remove_reference_t<Tp>&&>(tmp);
        }
      }

      template<typename Tp, typename Up, size_t Num, typename = std::void_t<
        decltype(std::declval<const swap_impl&>()(std::declval<Tp&>(), std::declval<Up&>()))>>
      constexpr void
      operator()(Tp (&e1)[Num], Up (&e2)[Num]) const noexcept(noexcept(std::declval<const swap_impl&>()(*e1, *e2)))
      {
        for (size_t n = 0; n < Num; ++n) (*this)(e1[n], e2[n]);
      }
    };
  } 


  inline constexpr detail_swap::swap_impl swap;
#endif
}


namespace OpenKalman::stdcompat
{
#if __cplusplus >= 202002L
  using std::swappable;
  using std::swappable_with;
#else
  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct swappable_impl : std::false_type {};

    template<typename T, typename U>
    struct swappable_impl<T, U, std::void_t<decltype(stdcompat::ranges::swap(std::declval<T&>(), std::declval<U&>()))>> : std::true_type {};

    template<typename T, typename U, typename = void>
    struct swappable_with_impl : std::false_type {};

    template<typename T, typename U>
    struct swappable_with_impl<T, U, std::void_t<decltype(stdcompat::ranges::swap(std::declval<T&&>(), std::declval<U&&>()))>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool swappable = detail::swappable_impl<T, T>::value;


  template<typename T, typename U>
  inline constexpr bool swappable_with =
    stdcompat::common_reference_with<T, U> and
    detail::swappable_with_impl<T, T>::value and
    detail::swappable_with_impl<U, U>::value and
    detail::swappable_with_impl<T, U>::value and
    detail::swappable_with_impl<U, T>::value;

#endif
}


#endif
