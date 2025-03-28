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
 * \brief Definition for \ref collections::internal::MovableWrapper.
 */

#ifndef OPENKALMAN_COLLECTIONS_MOVABLEWRAPPER_HPP
#define OPENKALMAN_COLLECTIONS_MOVABLEWRAPPER_HPP

#include <functional>

namespace OpenKalman::collections::internal
{
#if __cplusplus < 202002L
  namespace detail
  {
    template<typename T>
    struct reference_wrapper
    {
      template<typename U, std::enable_if_t<std::is_lvalue_reference_v<U> and
        not std::is_same_v<reference_wrapper, std::decay_t<U>>, int> = 0>
      constexpr reference_wrapper(U&& u) : ptr(std::addressof(std::forward<U>(u))) {}

      constexpr reference_wrapper(const reference_wrapper&) noexcept = default;

      constexpr reference_wrapper& operator=(const reference_wrapper& x) noexcept = default;

      constexpr T& get() const noexcept { return *ptr; }

    private:
      T* ptr;
    };
  }
#endif


  /**
   * \internal
   * \brief A movable wrapper that may contain either a value or a reference to a value.
   */
  template<typename T>
  struct MovableWrapper
  {
#if __cplusplus >= 202002L
    constexpr T& ref() { return t; }
    constexpr const T& ref() const { return t; }
    T t;
#else
    template<typename aT = T, std::enable_if_t<std::is_default_constructible_v<aT>, int> = 0>
    constexpr
    MovableWrapper() {}

    template<typename Arg, std::enable_if_t<std::is_constructible_v<T, Arg&&>, int> = 0>
    explicit constexpr MovableWrapper(Arg&& arg) : my_t {std::forward<Arg>(arg)} {}

    constexpr T& ref() { return my_t; }

    constexpr const T& ref() const { return my_t; }

  private:

    T my_t;
#endif
  };


  template<typename T>
  struct MovableWrapper<T&>
  {
#if __cplusplus >= 202002L
    std::reference_wrapper<T> t;
    constexpr T& ref() { return t.get(); }
    constexpr const T& ref() const { return t.get(); }
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<detail::reference_wrapper<T>, Arg&&>, int> = 0>
    explicit constexpr MovableWrapper(Arg&& arg) : my_t {std::forward<Arg>(arg)} {}

    constexpr T& ref() { return my_t.get(); }

    constexpr const T& ref() const { return my_t.get(); }

  private:

    detail::reference_wrapper<T> my_t;
#endif
  };

}

#endif //OPENKALMAN_COLLECTIONS_MOVABLEWRAPPER_HPP
