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
#include "basics/basics.hpp"

namespace OpenKalman::collections::internal
{
  /**
   * \internal
   * \brief A movable wrapper for any value, whether lvalue or rvalue.
   * \details The wrapper is guaranteed to be std::movable.
   * It is also guaranteed to be copyable if T is either copy_constructible or an lvalue reference.
   */
  template<typename T>
#ifdef __cpp_concepts
    requires (std::move_constructible<T> and std::is_object_v<T>) or std::is_lvalue_reference_v<T>
#endif
  struct movable_wrapper
  {
  private:

    static_assert(stdex::move_constructible<T> and std::is_object_v<T>);

    using T_ = OpenKalman::internal::movable_box<T>;

  public:

    /**
     * \brief The wrapped type
     */
    using type = T;


    /**
     * \brief Default constructor.
     */
    constexpr
#ifdef __cpp_lib_concepts
    movable_wrapper() noexcept(std::is_nothrow_default_constructible_v<T_>) requires std::default_initializable<T_> = default;
#else
    movable_wrapper() noexcept(std::is_nothrow_default_constructible_v<T_>) = default;
#endif


    /**
     * \brief Construct from an rvalue reference.
     */
    explicit constexpr
    movable_wrapper(T&& t) : t_ {std::move(t)} {}


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T& get() & noexcept { return t_.operator*(); }
    /// \overload
    constexpr const T& get() const & noexcept { return t_.operator*(); }
    /// \overload
    constexpr T&& get() && noexcept { return std::move(t_.operator*()); }
    /// \overload
    constexpr const T&& get() const && noexcept { return std::move(t_.operator*()); }


    /**
     * \brief Convert to the underlying value or its reference.
     */
    constexpr operator T& () & noexcept { return t_.operator*(); }
    /// \overload
    constexpr operator const T& () const & noexcept { return t_.operator*(); }
    /// \overload
    constexpr operator T () && noexcept { return std::move(t_.operator*()); }
    /// \overload
    constexpr operator const T () const && noexcept { return std::move(t_.operator*()); }


    /**
     * \brief Call the wrapped object if it is callable.
     */
    template<typename...ArgTypes>
    constexpr std::invoke_result_t<T&, ArgTypes...>
    operator () (ArgTypes&&...args) & noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
      return stdex::invoke(t_.operator*(), std::forward<ArgTypes>(args)...);
    }

    /// \overload
    template<typename...ArgTypes>
    std::invoke_result_t<const T&, ArgTypes...>
    constexpr operator () (ArgTypes&&...args) const & noexcept(std::is_nothrow_invocable_v<const T&, ArgTypes...>)
    {
      return stdex::invoke(t_.operator*(), std::forward<ArgTypes>(args)...);
    }

    /// \overload
    template<typename...ArgTypes>
    std::invoke_result_t<T&&, ArgTypes...>
    constexpr operator () (ArgTypes&&...args) && noexcept(std::is_nothrow_invocable_v<T&&, ArgTypes...>)
    {
      return stdex::invoke(std::move(t_.operator*()), std::forward<ArgTypes>(args)...);
    }

    /// \overload
    template<typename...ArgTypes>
    std::invoke_result_t<const T&&, ArgTypes...>
    constexpr operator () (ArgTypes&&...args) const && noexcept(std::is_nothrow_invocable_v<const T&&, ArgTypes...>)
    {
      return stdex::invoke(std::move(t_.operator*()), std::forward<ArgTypes>(args)...);
    }

  private:

    T_ t_;

  };


  /**
   * \overload
   * \internal
   * \brief Specialization for a reference type
   */
  template<typename T>
  struct movable_wrapper<T&>
  {
  private:

    using T_ = stdex::reference_wrapper<std::remove_reference_t<T>>;

  public:

    /**
     * \brief The wrapped type
     */
    using type = T&;


    /**
     * \brief Construct from an lvalue reference.
     */
    explicit constexpr
    movable_wrapper(T& t) : t_ {t} {}


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T& get() const noexcept { return t_; }


    /**
     * \brief Convert the wrapper to the underlying value.
     */
    constexpr operator T& () const noexcept { return t_; }


    /**
     * \brief Call the referenced object if it is callable.
     */
    template<typename...ArgTypes>
    constexpr std::invoke_result_t<T&, ArgTypes...>
    operator () (ArgTypes&&...args) const noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
      return stdex::invoke(t_, std::forward<ArgTypes>(args)...);
    }

  private:

    T_ t_;

  };


  /**
   * \brief deduction guide
   */
  template<typename T>
  movable_wrapper(T&&) -> movable_wrapper<T>;



#ifdef __cpp_impl_three_way_comparison
  template<typename T>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
    requires requires { {lhs.get() == rhs.get()} -> OpenKalman::internal::boolean_testable; }
  {
    return lhs.get() == rhs.get();
  }

  template<typename T>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
    requires (not std::is_const_v<T>) and requires { {lhs.get() == rhs.get()} -> OpenKalman::internal::boolean_testable; }
  {
    return lhs.get() == rhs.get();
  }

  template<typename T>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const T& ref) noexcept
    requires requires { {lhs.get() == ref} -> OpenKalman::internal::boolean_testable; }
  {
    return lhs.get() == ref;
  }

  template<typename T>
  constexpr auto operator<=>(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
    requires requires {
      {lhs.get() < rhs.get()} -> OpenKalman::internal::boolean_testable;
      {lhs.get() > rhs.get()} -> OpenKalman::internal::boolean_testable; }
  {
    return lhs.get() <=> rhs.get();
  }

  template<typename T>
  constexpr auto operator<=>(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept requires
    (not std::is_const_v<T>) and
    requires {
      {lhs.get() < rhs.get()} -> OpenKalman::internal::boolean_testable;
      {lhs.get() > rhs.get()} -> OpenKalman::internal::boolean_testable; }
  {
    return lhs.get() <=> rhs.get();
  }

  template<typename T>
  constexpr auto operator<=>(const movable_wrapper<T>& lhs, const T& ref) noexcept
    requires requires {
      {lhs.get() < ref} -> OpenKalman::internal::boolean_testable;
      {lhs.get() > ref} -> OpenKalman::internal::boolean_testable; }
  {
    return lhs.get() <=> ref;
  }
#else
  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() == rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() == rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() == std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator==(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() == rhs.get(); }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() == std::declval<const T&>())>, int> = 0>
  constexpr bool operator==(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() == ref; }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const T&>() == std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator==(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref == lhs.get(); }


  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator!=(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() != rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() != std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator!=(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() != rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() != std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator!=(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() != rhs.get(); }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() != std::declval<const T&>())>, int> = 0>
  constexpr bool operator!=(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() != ref; }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const T&>() != std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator!=(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref != lhs.get(); }


template<typename T, std::enable_if_t<
  OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get())>, int> = 0>
constexpr bool operator<(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() < rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() < std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator<(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() < rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() < std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator<(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() < rhs.get(); }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() < std::declval<const T&>())>, int> = 0>
  constexpr bool operator<(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() < ref; }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const T&>() < std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator<(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref < lhs.get(); }


template<typename T, std::enable_if_t<
  OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get())>, int> = 0>
constexpr bool operator>(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() > rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() > std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator>(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() > rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() > std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator>(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() > rhs.get(); }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() > std::declval<const T&>())>, int> = 0>
  constexpr bool operator>(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() > ref; }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const T&>() > std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator>(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref > lhs.get(); }


template<typename T, std::enable_if_t<
  OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get())>, int> = 0>
constexpr bool operator<=(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() <= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() <= std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator<=(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() <= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() <= std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator<=(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() <= rhs.get(); }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() <= std::declval<const T&>())>, int> = 0>
  constexpr bool operator<=(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() <= ref; }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const T&>() <= std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator<=(const T& ref, const movable_wrapper<T>& lhs) noexcept
  { return ref <= lhs.get(); }


template<typename T, std::enable_if_t<
  OpenKalman::internal::boolean_testable<decltype(std::declval<const movable_wrapper<T>&>().get() == std::declval<const movable_wrapper<T>&>().get())>, int> = 0>
constexpr bool operator>=(const movable_wrapper<T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() >= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() >= std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator>=(const movable_wrapper<T>& lhs, const movable_wrapper<const T>& rhs) noexcept
  { return lhs.get() >= rhs.get(); }

  template<typename T, std::enable_if_t<(not std::is_const_v<T>) and
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() >= std::declval<movable_wrapper<T>&>().get())>, int> = 0>
  constexpr bool operator>=(const movable_wrapper<const T>& lhs, const movable_wrapper<T>& rhs) noexcept
  { return lhs.get() >= rhs.get(); }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<movable_wrapper<T>&>().get() >= std::declval<const T&>())>, int> = 0>
  constexpr bool operator>=(const movable_wrapper<T>& lhs, const T& ref) noexcept
  { return lhs.get() >= ref; }

  template<typename T, std::enable_if_t<
    OpenKalman::internal::boolean_testable<decltype(std::declval<const T&>() >= std::declval<movable_wrapper<T>&>().get())>, int> = 0>
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
    requires { typename stdex::common_reference_t<typename R::type, TQ>; } and
    std::convertible_to<RQ, stdex::common_reference_t<typename R::type, TQ>>
  ;
#endif

}


namespace std
{
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


}

#endif
