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
#include "basics/language-features.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief A movable wrapper for any value, whether lvalue or rvalue.
   * \details The wrapper is copyable only if it wraps an underlying lvalue.
   */
  template<typename T>
  struct movable_wrapper
  {
    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    movable_wrapper() noexcept requires std::default_initializable<T> = default;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr movable_wrapper() noexcept {}
#endif


    /**
     * \brief Construct from a value.
     */
    explicit constexpr
    movable_wrapper(T&& t) noexcept : my_t {std::forward<T>(t)} {}


    /**
     * \brief Move constructor.
     * \note The copy constructor is implicitly deleted in this specialization.
     */
    constexpr
    movable_wrapper(movable_wrapper&& arg) = default;


    /**
     * \brief Move assignment operator.
     * \note The copy assignment operator is implicitly deleted in this specialization.
     */
    constexpr
    movable_wrapper& operator=(movable_wrapper&& other) = default;


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T& get() & noexcept { return my_t; }
    /// \overload
    constexpr const T& get() const & noexcept { return my_t; }
    /// \overload
    constexpr T&& get() && noexcept { return std::move(my_t); }
    /// \overload
    constexpr const T&& get() const && noexcept { return std::move(my_t); }


    /**
     * \brief Convert the wrapper to the underlying value.
     */
    constexpr operator T& () & noexcept { return my_t; }
    /// \overload
    constexpr operator const T& () const & noexcept { return my_t; }
    /// \overload
    constexpr operator T&& () && noexcept { return std::move(my_t); }
    /// \overload
    constexpr operator const T&& () const && noexcept { return std::move(my_t); }

  private:

    T my_t;

  };


  /**
   * \overload
   */
  template<typename T>
  struct movable_wrapper<T&>
  {
  private:

#if __cplusplus >= 202002L
    using MyT = std::reference_wrapper<std::remove_reference_t<T>>;
#else
    using MyT = reference_wrapper<std::remove_reference_t<T>>;
#endif

  public:

    /**
     * \brief Construct from a value.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    explicit constexpr
    movable_wrapper(Arg&& arg) noexcept requires std::constructible_from<MyT, Arg&&>
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<MyT, Arg&&>, int> = 0>
    explicit constexpr movable_wrapper(Arg&& arg) noexcept
#endif
      : my_t {std::forward<Arg>(arg)} {}


    /**
     * \brief Retrieve the stored value.
     */
    constexpr T& get() const noexcept { return my_t; }


    /**
     * \brief Convert the wrapper to the underlying value.
     */
    constexpr operator T& () const noexcept { return my_t; }

  private:

    MyT my_t;

  };


  /**
   * \brief deduction guide
   */
  template<typename T>
  movable_wrapper(T&&) -> movable_wrapper<T>;


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

}

#endif //OPENKALMAN_MOVABLE_WRAPPER_HPP
