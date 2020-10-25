/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_ELEMENTSETTER_HPP
#define OPENKALMAN_ELEMENTSETTER_HPP

#include <functional>

namespace OpenKalman::internal
{
  template<typename T>
  struct ElementSetter<false, T>
  {
    using Scalar = typename MatrixTraits<T>::Scalar;
    using OnChange = std::function<void()>;
    using OnSet = std::function<void()>;

    ElementSetter(T& t, std::size_t i, std::size_t j, const OnSet& on_change = []{}, const OnChange& on_set = []{})
      : getter([i, j, &t, on_change] () -> Scalar
        {
          on_change();
          return get_element(t, i, j);
        }),
        setter([i, j, &t, on_set, on_change] (Scalar s)
        {
          on_change();
          set_element(t, s, i, j);
          on_set();
        })
      {
        static_assert(is_element_gettable_v<T, 2>, "Two-index element read access is not available.");
        static_assert(is_element_settable_v<T, 2>, "Two-index element write access is not available.");
      }

    ElementSetter(T& t, std::size_t i, const OnSet& on_change = []{}, const OnChange& on_set = []{})
      : getter([i, &t, on_change] () -> Scalar
        {
          on_change();
          return get_element(t, i);
        }),
        setter([i, &t, on_set, on_change] (Scalar s)
        {
          on_change();
          set_element(t, s, i);
          on_set();
        })
      {
        static_assert(is_element_gettable_v<T, 1>, "One-index element read access is not available.");
        static_assert(is_element_settable_v<T, 1>, "One-index element write access is not available.");
      }

    /// Get an element.
    operator Scalar() const { return getter(); }

    /// Set an element.
    void operator=(Scalar s) { setter(s); }

  protected:
    const std::function<Scalar()> getter;
    const std::function<void(Scalar)> setter;
  };


  template<typename T>
  struct ElementSetter<true, T>
  {
    using Scalar = typename MatrixTraits<T>::Scalar;
    using OnChange = std::function<void()>;

    ElementSetter(const T& t, std::size_t i, std::size_t j, const OnChange& on_change = []{})
      : getter([i, j, &t, on_change] () -> Scalar
        {
          on_change();
          return get_element(t, i, j);
        })
      {
        static_assert(is_element_gettable_v<T, 2>, "Two-index element read access is not available.");
      }

    ElementSetter(const T& t, std::size_t i, const OnChange& on_change = []{})
      : getter([i, &t, on_change] () -> Scalar
        {
          on_change();
          return get_element(t, i);
        })
      {
        static_assert(is_element_gettable_v<T, 1>, "One-index element read access is not available.");
      }

    /// Get an element.
    operator Scalar() const { return getter(); }

  protected:
    const std::function<Scalar()> getter;
  };


  template<typename I1, typename I2, typename T, typename OnSet, typename OnChange, std::enable_if_t<
    is_element_gettable_v<T, 2> and
    std::is_integral_v<std::decay_t<I1>> and std::is_integral_v<std::decay_t<I2>> and
    std::is_invocable_r_v<void, OnSet> and
    std::is_invocable_r_v<void, OnChange>, int> = 0>
  ElementSetter(T, I1, I2, OnChange, OnSet) -> ElementSetter<false, std::decay_t<T>>;

  template<typename I, typename T, typename OnSet, typename OnChange, std::enable_if_t<
    is_element_gettable_v<T, 1> and
    std::is_integral_v<std::decay_t<I>> and
    std::is_invocable_r_v<void, OnSet> and
    std::is_invocable_r_v<void, OnChange>, int> = 0>
  ElementSetter(T, I, OnChange, OnSet) -> ElementSetter<false, std::decay_t<T>>;

  template<typename I1, typename I2, typename T, typename X, std::enable_if_t<
    is_element_gettable_v<T, 2> and
    std::is_integral_v<std::decay_t<I1>> and std::is_integral_v<std::decay_t<I2>> and
    std::is_invocable_r_v<void, X>, int> = 0>
  ElementSetter(T, I1, I2, X)
    -> ElementSetter<not is_element_settable_v<T, 2>, std::decay_t<T>>;

  template<typename I, typename T, typename X, std::enable_if_t<
    is_element_gettable_v<T, 1> and
    std::is_integral_v<std::decay_t<I>> and not std::is_integral_v<std::decay_t<X>> and
    std::is_invocable_r_v<void, X>, int> = 0>
  ElementSetter(T, I, X) -> ElementSetter<not is_element_settable_v<T, 1>, std::decay_t<T>>;

  template<typename I1, typename I2, typename T, std::enable_if_t<
    (true or is_element_gettable_v<T, 2>) and
    std::is_integral_v<std::decay_t<I1>> and std::is_integral_v<std::decay_t<I2>>, int> = 0>
  ElementSetter(T, I1, I2) -> ElementSetter<not is_element_settable_v<T, 2>, std::decay_t<T>>;

  template<typename I, typename T, std::enable_if_t<
    is_element_gettable_v<T, 1> and
    std::is_integral_v<std::decay_t<I>>, int> = 0>
  ElementSetter(T, I) -> ElementSetter<not is_element_settable_v<T, 1>, std::decay_t<T>>;


  template<bool read_only, typename T>
  auto make_ElementSetter(
    T&& t,
    std::size_t i,
    std::size_t j,
    const std::function<void()>& on_change,
    const std::function<void()>& on_set)
  {
    if constexpr(read_only)
      return ElementSetter<true, std::decay_t<T>>(std::forward<T>(t), i, j, on_change);
    else
      return ElementSetter<false, std::decay_t<T>>(std::forward<T>(t), i, j, on_change, on_set);
  }


  template<bool read_only, typename T>
  auto make_ElementSetter(
    T&& t,
    std::size_t i,
    const std::function<void()>& on_change,
    const std::function<void()>& on_set)
  {
    if constexpr(read_only)
      return ElementSetter<true, std::decay_t<T>>(std::forward<T>(t), i, on_change);
    else
      return ElementSetter<false, std::decay_t<T>>(std::forward<T>(t), i, on_change, on_set);
  }

}

#endif //OPENKALMAN_ELEMENTSETTER_HPP
