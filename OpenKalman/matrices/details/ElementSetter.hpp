/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_ELEMENTSETTER_HPP
#define OPENKALMAN_ELEMENTSETTER_HPP

#include <functional>
#include <mutex>

namespace OpenKalman::internal
{
  template<typename T>
  struct ElementSetter<false, T>
  {
    using Scalar = typename MatrixTraits<T>::Scalar;
    using BeforeAccess = std::function<void()>;
    using AfterSet = std::function<void()>;

    ElementSetter(T& t, std::size_t i, std::size_t j,
      const BeforeAccess& before_access = []{}, const AfterSet& after_set = []{})
      : getter([i, j, &t, before_access] () -> Scalar
        {
          before_access();
          return get_element(t, i, j);
        }),
        setter([i, j, &t, after_set, before_access] (Scalar s)
        {
          before_access();
          set_element(t, s, i, j);
          after_set();
        })
      {
        static_assert(element_gettable<T, 2>, "Two-index element read access is not available.");
        static_assert(element_settable<T, 2>, "Two-index element write access is not available.");
      }

    ElementSetter(T& t, std::size_t i, const BeforeAccess& before_access = []{}, const AfterSet& after_set = []{})
      : getter([i, &t, before_access] () -> Scalar
        {
          before_access();
          return get_element(t, i);
        }),
        setter([i, &t, after_set, before_access] (Scalar s)
        {
          before_access();
          set_element(t, s, i);
          after_set();
        })
      {
        static_assert(element_gettable<T, 1>, "One-index element read access is not available.");
        static_assert(element_settable<T, 1>, "One-index element write access is not available.");
      }

    /// Get an element.
    operator Scalar() const { std::scoped_lock lock {setter_mutex}; return getter(); }

    /// Set an element.
    void operator=(Scalar s) { std::scoped_lock lock {setter_mutex}; setter(s); }

  private:
    const std::function<Scalar()> getter;

    const std::function<void(Scalar)> setter;

    mutable std::mutex setter_mutex;

  };


  template<typename T>
  struct ElementSetter<true, T>
  {
    using Scalar = typename MatrixTraits<T>::Scalar;
    using BeforeAccess = std::function<void()>;

    ElementSetter(const T& t, std::size_t i, std::size_t j, const BeforeAccess& before_access = []{})
      : getter([i, j, &t, before_access] () -> Scalar
        {
          before_access();
          return get_element(t, i, j);
        })
      {
        static_assert(element_gettable<T, 2>, "Two-index element read access is not available.");
      }

    ElementSetter(const T& t, std::size_t i, const BeforeAccess& before_access = []{})
      : getter([i, &t, before_access] () -> Scalar
        {
          before_access();
          return get_element(t, i);
        })
      {
        static_assert(element_gettable<T, 1>, "One-index element read access is not available.");
      }

    /// Get an element.
    operator Scalar() const { std::scoped_lock lock {setter_mutex}; return getter(); }

  private:
    const std::function<Scalar()> getter;

    mutable std::mutex setter_mutex;

  };


  template<typename I1, typename I2, typename T, typename BeforeAccess, typename AfterSet, std::enable_if_t<
    element_gettable<T, 2> and
    std::is_integral_v<std::decay_t<I1>> and std::is_integral_v<std::decay_t<I2>> and
    std::is_invocable_r_v<void, BeforeAccess> and
    std::is_invocable_r_v<void, AfterSet>, int> = 0>
  ElementSetter(T, I1, I2, BeforeAccess, AfterSet) -> ElementSetter<false, std::decay_t<T>>;

  template<typename I, typename T, typename BeforeAccess, typename AfterSet, std::enable_if_t<
    element_gettable<T, 1> and
    std::is_integral_v<std::decay_t<I>> and
    std::is_invocable_r_v<void, BeforeAccess> and
    std::is_invocable_r_v<void, AfterSet>, int> = 0>
  ElementSetter(T, I, BeforeAccess, AfterSet) -> ElementSetter<false, std::decay_t<T>>;

  template<typename I1, typename I2, typename T, typename X, std::enable_if_t<
    element_gettable<T, 2> and
    std::is_integral_v<std::decay_t<I1>> and std::is_integral_v<std::decay_t<I2>> and
    std::is_invocable_r_v<void, X>, int> = 0>
  ElementSetter(T, I1, I2, X)
    -> ElementSetter<not element_settable<T, 2>, std::decay_t<T>>;

  template<typename I, typename T, typename X, std::enable_if_t<
    element_gettable<T, 1> and
    std::is_integral_v<std::decay_t<I>> and not std::is_integral_v<std::decay_t<X>> and
    std::is_invocable_r_v<void, X>, int> = 0>
  ElementSetter(T, I, X) -> ElementSetter<not element_settable<T, 1>, std::decay_t<T>>;

  template<typename I1, typename I2, typename T, std::enable_if_t<
    (true or element_gettable<T, 2>) and
    std::is_integral_v<std::decay_t<I1>> and std::is_integral_v<std::decay_t<I2>>, int> = 0>
  ElementSetter(T, I1, I2) -> ElementSetter<not element_settable<T, 2>, std::decay_t<T>>;

  template<typename I, typename T, std::enable_if_t<
    element_gettable<T, 1> and
    std::is_integral_v<std::decay_t<I>>, int> = 0>
  ElementSetter(T, I) -> ElementSetter<not element_settable<T, 1>, std::decay_t<T>>;


  template<bool read_only, typename T>
  auto make_ElementSetter(
    T&& t,
    std::size_t i,
    std::size_t j,
    const std::function<void()>& before_access, // defaults to []{}
    const std::function<void()>& after_set) // defaults to []{}
  {
    if constexpr(read_only)
      return ElementSetter<true, std::decay_t<T>>(std::forward<T>(t), i, j, before_access);
    else
      return ElementSetter<false, std::decay_t<T>>(std::forward<T>(t), i, j, before_access, after_set);
  }


  template<bool read_only, typename T>
  auto make_ElementSetter(
    T&& t,
    std::size_t i,
    const std::function<void()>& before_access, // defaults to []{}
    const std::function<void()>& after_set) // defaults to []{}
  {
    if constexpr(read_only)
      return ElementSetter<true, std::decay_t<T>>(std::forward<T>(t), i, before_access);
    else
      return ElementSetter<false, std::decay_t<T>>(std::forward<T>(t), i, before_access, after_set);
  }

}

#endif //OPENKALMAN_ELEMENTSETTER_HPP
