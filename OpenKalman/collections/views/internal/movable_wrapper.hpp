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
 * \brief Definition for \ref internal::movable_wrapper.
 */

#ifndef OPENKALMAN_MOVABLE_WRAPPER_HPP
#define OPENKALMAN_MOVABLE_WRAPPER_HPP

#include <functional>
#include "basics/compatibility/language-features.hpp"
#include "collections/concepts/tuple_like.hpp"
#ifndef __cpp_concepts
#include "maybe_tuple_size.hpp"
#include "maybe_tuple_element.hpp"
#endif

namespace OpenKalman::collections::internal
{
  //
  // -- General specialization -- //
  //

  /**
   * \internal
   * \brief A movable wrapper for any value, whether lvalue or rvalue.
   * \details The wrapper is copyable only if it wraps an underlying lvalue.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct movable_wrapper
  {
    /**
     * \brief The wrapped type
     */
    using type = T;


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    movable_wrapper() noexcept = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr movable_wrapper() noexcept {}
#endif


    /**
     * \brief Construct from a value.
     */
    explicit constexpr
    movable_wrapper(T&& t) noexcept : t_ {std::forward<T>(t)} {}


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T& get() & noexcept { return t_; }
    /// \overload
    constexpr const T& get() const & noexcept { return t_; }
    /// \overload
    constexpr T&& get() && noexcept { return std::move(t_); }
    /// \overload
    constexpr const T&& get() const && noexcept { return std::move(t_); }


    /**
     * \brief Convert to the underlying value or its reference.
     */
    constexpr operator T& () & noexcept { return t_; }
    /// \overload
    constexpr operator const T& () const & noexcept { return t_; }
    /// \overload
    constexpr operator T () && noexcept { return std::move(t_); }
    /// \overload
    constexpr operator const T () const && noexcept { return std::move(t_); }


    /**
     * \brief Call the callable object.
     */
    template<typename...ArgTypes>
    constexpr std::invoke_result_t<T&, ArgTypes...>
    operator () (ArgTypes&&...args) & noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
#if __cplusplus >= 202002L
      return std::invoke(t_, std::forward<ArgTypes>(args)...);
#else
      return OpenKalman::invoke(t_, std::forward<ArgTypes>(args)...);
#endif
    }

    /// \overload
    template<typename...ArgTypes>
    std::invoke_result_t<T&, ArgTypes...>
    constexpr operator () (ArgTypes&&...args) const & noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
#if __cplusplus >= 202002L
      return std::invoke(t_, std::forward<ArgTypes>(args)...);
#else
      return OpenKalman::invoke(t_, std::forward<ArgTypes>(args)...);
#endif
    }

    /// \overload
    template<typename...ArgTypes>
    std::invoke_result_t<T&, ArgTypes...>
    constexpr operator () (ArgTypes&&...args) && noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
#if __cplusplus >= 202002L
      return std::invoke(std::move(t_), std::forward<ArgTypes>(args)...);
#else
      return OpenKalman::invoke(std::move(t_), std::forward<ArgTypes>(args)...);
#endif
    }

    /// \overload
    template<typename...ArgTypes>
    std::invoke_result_t<T&, ArgTypes...>
    constexpr operator () (ArgTypes&&...args) const && noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
#if __cplusplus >= 202002L
      return std::invoke(std::move(t_), std::forward<ArgTypes>(args)...);
#else
      return OpenKalman::invoke(std::move(t_), std::forward<ArgTypes>(args)...);
#endif
    }

  private:

    T t_;

  };


  //
  // -- Reference specialization -- //
  //

  /**
   * \overload
   */
  template<typename T>
  struct movable_wrapper<T&>
  {
  private:

#if __cplusplus >= 202002L
    using T_ = std::reference_wrapper<std::remove_reference_t<T>>;
#else
    using T_ = reference_wrapper<std::remove_reference_t<T>>;
#endif

  public:

    /**
     * \brief The wrapped type
     */
    using type = T&;


    /**
     * \brief Construct from a value.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    explicit constexpr
    movable_wrapper(Arg&& arg) noexcept requires std::constructible_from<T_, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<T_, Arg&&>, int> = 0>
    explicit constexpr movable_wrapper(Arg&& arg) noexcept
#endif
      : t_ {std::forward<Arg>(arg)} {}


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T& get() const noexcept { return t_; }


    /**
     * \brief Convert the wrapper to the underlying value.
     */
    constexpr operator T& () const noexcept { return t_; }


    /**
     * \brief Call the callable object.
     */
    template<typename...ArgTypes>
    constexpr std::invoke_result_t<T&, ArgTypes...>
    operator () (ArgTypes&&...args) const noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
      return t_(std::forward<ArgTypes>(args)...);
    }

  private:

    T_ t_;

  };


  //
  // -- Tuple specialization -- //
  //

  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<tuple_like T> requires (not std::movable<T>)
  struct movable_wrapper<T>
#else
  template<typename T>
  struct movable_wrapper<T, std::enable_if_t<not std::is_lvalue_reference_v<T> and tuple_like<T> and not movable<T>>>
#endif
  {
  private:

    template<typename U>
    static constexpr decltype(auto)
    make_element(U&& u)
    {
      if constexpr (std::is_lvalue_reference_v<U>) return movable_wrapper<U> {u};
      else return std::forward<U>(u);
    }


    template<typename U, std::size_t...i>
    static constexpr auto
    fill_tuple(U&& u, std::index_sequence<i...>)
    {
      return std::tuple { make_element(OpenKalman::internal::generalized_std_get<i>(std::forward<U>(u)))...};
    }


    using T_ = decltype(fill_tuple(std::declval<T>(), std::make_index_sequence<std::tuple_size_v<T>>{}));

  public:

    /**
     * \brief The wrapped type
     */
    using type = T;


    /**
     * \brief Construct from a tuple-like object.
     */
    explicit constexpr
    movable_wrapper(T&& t) noexcept
      : t_ {fill_tuple(std::forward<T>(t), std::make_index_sequence<std::tuple_size_v<T>>{})} {}


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T_& get() & noexcept { return t_; }
    /// \overload
    constexpr const T_& get() const & noexcept { return t_; }
    /// \overload
    constexpr T_&& get() && noexcept { return std::move(t_); }
    /// \overload
    constexpr const T_&& get() const && noexcept { return std::move(t_); }

  private:

    T_ t_;

  };


  /**
   * \brief deduction guide
   */
  template<typename T>
  movable_wrapper(T&&) -> movable_wrapper<T>;


  /**
   * \brief If a \ref movable_wrapper is \ref tuple_like, get its element i.
   * \tparam i An index to the element
   */
#ifdef __cpp_concepts
  template<std::size_t i, typename T> requires tuple_like<T>
#else
  template<std::size_t i, typename T, std::enable_if_t<tuple_like<T>, int> = 0>
#endif
  constexpr decltype(auto)
  get(movable_wrapper<T>& arg) noexcept
  {
    static_assert(i < std::tuple_size_v<std::decay_t<T>>, "Index out of range");
    return OpenKalman::internal::generalized_std_get<i>(arg.get());
  }


#ifdef __cpp_concepts
  template<std::size_t i, typename T> requires tuple_like<T>
#else
  template<std::size_t i, typename T, std::enable_if_t<tuple_like<T>, int> = 0>
#endif
  constexpr decltype(auto)
  get(const movable_wrapper<T>& arg) noexcept
  {
    static_assert(i < std::tuple_size_v<std::decay_t<T>>, "Index out of range");
    return OpenKalman::internal::generalized_std_get<i>(arg.get());
  }


#ifdef __cpp_concepts
  template<std::size_t i, typename T> requires tuple_like<T>
#else
  template<std::size_t i, typename T, std::enable_if_t<tuple_like<T>, int> = 0>
#endif
  constexpr decltype(auto)
  get(movable_wrapper<T>&& arg) noexcept
  {
    static_assert(i < std::tuple_size_v<std::decay_t<T>>, "Index out of range");
    return OpenKalman::internal::generalized_std_get<i>(std::move(arg).get());
  }


#ifdef __cpp_concepts
  template<std::size_t i, typename T> requires tuple_like<T>
#else
  template<std::size_t i, typename T, std::enable_if_t<tuple_like<T>, int> = 0>
#endif
  constexpr decltype(auto)
  get(const movable_wrapper<T>&& arg) noexcept
  {
    static_assert(i < std::tuple_size_v<std::decay_t<T>>, "Index out of range");
    return OpenKalman::internal::generalized_std_get<i>(std::move(arg).get());
  }


#ifdef __cpp_impl_three_way_comparison
  template<typename T>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
    requires requires { {lhs.get() == rhs.get()} -> std::convertible_to<bool>; }
  {
    return lhs.get() == rhs.get();
  }

  template<typename T>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
    requires (not std::is_const_v<T>) and requires { {lhs.get() == rhs.get()} -> std::convertible_to<bool>; }
  {
    return lhs.get() == rhs.get();
  }

  template<typename T>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const T& ref) noexcept
    requires requires { {lhs.get() == ref} -> std::convertible_to<bool>; }
  {
    return lhs.get() == ref;
  }

  template<typename T>
  constexpr auto operator<=>(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
    requires requires { {lhs.get() < rhs.get()} -> std::convertible_to<bool>; {lhs.get() > rhs.get()} -> std::convertible_to<bool>; }
  {
    return lhs.get() <=> rhs.get();
  }

  template<typename T>
  constexpr auto operator<=>(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept requires
    (not std::is_const_v<T>) and
    requires { {lhs.get() < rhs.get()} -> std::convertible_to<bool>; {lhs.get() > rhs.get()} -> std::convertible_to<bool>; }
  {
    return lhs.get() <=> rhs.get();
  }

  template<typename T>
  constexpr auto operator<=>(const movable_wrapper<T>& lhs, const T& ref) noexcept
    requires requires { {lhs.get() < ref} -> std::convertible_to<bool>; {lhs.get() > ref} -> std::convertible_to<bool>; }
  {
    return lhs.get() <=> ref;
  }
#else
  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() == rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() == rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() == std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator==(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() == rhs.get(); }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() == std::declval<const T&>()), bool>::value, int> = 0>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() == ref; }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const T&>() == std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator==(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref == lhs.get(); }


  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator!=(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() != rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() != std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator!=(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() != rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() != std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator!=(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() != rhs.get(); }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() != std::declval<const T&>()), bool>::value, int> = 0>
  constexpr bool operator!=(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() != ref; }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const T&>() != std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator!=(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref != lhs.get(); }


template<typename T, std::enable_if_t<
  std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get()), bool>::value, int> = 0>
constexpr bool operator<(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() < rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() < std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator<(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() < rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() < std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator<(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() < rhs.get(); }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() < std::declval<const T&>()), bool>::value, int> = 0>
  constexpr bool operator<(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() < ref; }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const T&>() < std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator<(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref < lhs.get(); }


template<typename T, std::enable_if_t<
  std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get()), bool>::value, int> = 0>
constexpr bool operator>(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() > rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() > std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator>(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() > rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() > std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator>(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() > rhs.get(); }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() > std::declval<const T&>()), bool>::value, int> = 0>
  constexpr bool operator>(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() > ref; }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const T&>() > std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator>(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref > lhs.get(); }


template<typename T, std::enable_if_t<
  std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get()), bool>::value, int> = 0>
constexpr bool operator<=(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() <= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() <= std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator<=(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() <= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() <= std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator<=(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() <= rhs.get(); }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() <= std::declval<const T&>()), bool>::value, int> = 0>
  constexpr bool operator<=(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() <= ref; }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const T&>() <= std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator<=(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref <= lhs.get(); }


template<typename T, std::enable_if_t<
  std::is_convertible<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get()), bool>::value, int> = 0>
constexpr bool operator>=(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() >= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() >= std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator>=(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() >= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() >= std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator>=(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() >= rhs.get(); }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<movable_wrapper<T>&>().get() >= std::declval<const T&>()), bool>::value, int> = 0>
  constexpr bool operator>=(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() >= ref; }

  template<typename T, std::enable_if_t<
    std::is_convertible<decltype(std::declval<const T&>() >= std::declval<movable_wrapper<T>&>().get()), bool>::value, int> = 0>
  constexpr bool operator>=(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref >= lhs.get(); }
#endif


#if __cplusplus >= 202002L
  namespace detail
  {
    template <class T>
    inline constexpr bool is_movable_wrapper_ref = false;

    template <class T>
    inline constexpr bool is_movable_wrapper_ref<movable_wrapper<T&>> = true;
  }


  template<typename R, typename T, typename RQ, typename TQ>
  concept movable_wrapper_common_reference_exists_with =
    detail::is_movable_wrapper_ref<R> and
    requires { typename std::common_reference_t<typename R::type, TQ>; } and
    std::convertible_to<RQ, std::common_reference_t<typename R::type, TQ>>
  ;
#endif

}


namespace std
{
#ifdef __cpp_concepts
  template<typename Tup> requires requires { tuple_size<std::decay_t<Tup>>::value; }
  struct tuple_size<OpenKalman::collections::internal::movable_wrapper<Tup>> : tuple_size<std::decay_t<Tup>> {};
#else
  template<typename Tup>
  struct tuple_size<OpenKalman::collections::internal::movable_wrapper<Tup>>
    : OpenKalman::collections::internal::maybe_tuple_size<std::decay_t<Tup>> {};
#endif


#ifdef __cpp_concepts
  template<std::size_t i, typename Tup> requires requires { typename tuple_element<i, std::decay_t<Tup>>::type; }
  struct tuple_element<i, OpenKalman::collections::internal::movable_wrapper<Tup>> : tuple_element<i, std::decay_t<Tup>> {};
#else
  template<std::size_t i, typename Tup>
  struct tuple_element<i, OpenKalman::collections::internal::movable_wrapper<Tup>>
    : OpenKalman::collections::internal::maybe_tuple_element<i, std::decay_t<Tup>> {};
#endif


#if __cplusplus >= 202002L
  template <typename R, typename T, template<typename> typename RQual,  template<typename> typename TQual> requires
    OpenKalman::collections::internal::movable_wrapper_common_reference_exists_with<R, T, RQual<R>, TQual<T>> and
    (not OpenKalman::collections::internal::movable_wrapper_common_reference_exists_with<T, R, TQual<T>, RQual<R>>)
  struct basic_common_reference<R, T, RQual, TQual>
  {
    using type = common_reference_t<typename R::type, TQual<T>>;
  };


  template <typename T, typename R, template <typename> typename TQual,  template <typename> typename RQual> requires
    OpenKalman::collections::internal::movable_wrapper_common_reference_exists_with<R, T, RQual<R>, TQual<T>> and
    (not OpenKalman::collections::internal::movable_wrapper_common_reference_exists_with<T, R, TQual<T>, RQual<R>>)
  struct basic_common_reference<T, R, TQual, RQual>
  {
    using type = common_reference_t<typename R::type, TQual<T>>;
  };
#endif


} // namespace std

#endif //OPENKALMAN_MOVABLE_WRAPPER_HPP
