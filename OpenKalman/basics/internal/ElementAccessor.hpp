/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of OpenKalman::internal::ElementAccessor
 */

#ifndef OPENKALMAN_ELEMENTACCESSOR_HPP
#define OPENKALMAN_ELEMENTACCESSOR_HPP

#include <functional>

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Settable ElementAccessor specialization.
   * \todo Add a template parameter for number of coefficients, and provide distinct template instances.
   */
  template<typename Scalar>
  struct ElementAccessor<true, Scalar>
  {

#ifdef __cpp_concepts
    ElementAccessor(element_settable<2> auto& t, std::size_t i, std::size_t j,
        std::invocable auto&& pre_access = []{}, std::invocable auto&& post_set = []{})
      requires std::same_as<Scalar, scalar_type_of_t<decltype(t)>> and (not std::is_const_v<decltype(t)>) and
        element_gettable<decltype(t), 2>
#else
    template<typename T, typename PreAccess = std::function<void()>, typename PostSet = std::function<void()>,
        std::enable_if_t<
      std::is_constructible_v<std::function<void()>, PreAccess&&> and
      std::is_constructible_v<std::function<void()>, PostSet&&> and
      std::is_same_v<Scalar, typename scalar_type_of<T>::type> and
      (not std::is_const_v<T>) and element_gettable<T&, 2> and element_settable<T&, 2>, int> = 0>
    ElementAccessor(T& t, std::size_t i, std::size_t j, PreAccess&& pre_access = []{}, PostSet&& post_set = []{})
#endif
      : before_access {std::forward<decltype(pre_access)>(pre_access)},
        getter {[&t, i, j] () -> Scalar { return get_element(t, i, j); }},
        setter {[&t, i, j] (Scalar s) {set_element(t, s, i, j); }},
        after_set {std::forward<decltype(post_set)>(post_set)} {}


#ifdef __cpp_concepts
    ElementAccessor(element_settable<1> auto& t, std::size_t i,
        std::invocable auto&& pre_access = []{}, std::invocable auto&& post_set = []{})
      requires std::same_as<Scalar, scalar_type_of_t<decltype(t)>> and (not std::is_const_v<decltype(t)>) and
        element_gettable<decltype(t), 1>
#else
    template<typename T, typename PreAccess = std::function<void()>, typename PostSet = std::function<void()>,
        std::enable_if_t<
      std::is_constructible_v<std::function<void()>, PreAccess&&> and std::is_invocable_v<PostSet> and
      std::is_constructible_v<std::function<void()>, PostSet&&> and
      std::is_same_v<Scalar, typename scalar_type_of<T>::type> and
      (not std::is_const_v<T>) and element_gettable<T&, 1> and element_settable<T&, 1>, int> = 0>
    ElementAccessor(T& t, std::size_t i, PreAccess&& pre_access = []{}, PostSet&& post_set = []{})
#endif
      : before_access {std::forward<decltype(pre_access)>(pre_access)},
        getter {[&t, i] () -> Scalar { return get_element(t, i); }},
        setter {[&t, i] (Scalar s) { set_element(t, s, i); }},
        after_set {std::forward<decltype(post_set)>(post_set)} {}


    /// Get an element.
    operator Scalar() const
    {
      before_access();
      return getter();
    }


    /// Set an element.
    void operator=(Scalar s)
    {
      before_access();
      setter(s);
      after_set();
    }

  private:

    const std::function<void()> before_access;

    const std::function<Scalar()> getter;

    const std::function<void(Scalar)> setter;

    const std::function<void()> after_set;

  };


  /**
   * \internal
   * \brief Read-only ElementAccessor specialization.
   */
  template<typename Scalar>
  struct ElementAccessor<false, Scalar>
  {

#ifdef __cpp_concepts
    ElementAccessor(const element_gettable<2> auto& t, std::size_t i, std::size_t j, std::invocable auto&& pre_access = []{})
      requires std::same_as<Scalar, scalar_type_of_t<decltype(t)>>
#else
    template<typename T, typename PreAccess = std::function<void()>, std::enable_if_t<
      std::is_constructible_v<std::function<void()>, PreAccess&&> and
      std::is_same_v<Scalar, typename scalar_type_of<T>::type> and element_gettable<const T&, 2>, int> = 0>
    ElementAccessor(const T& t, std::size_t i, std::size_t j, PreAccess&& pre_access = []{})
#endif
      : before_access {std::forward<decltype(pre_access)>(pre_access)},
        getter {[&t, i, j] () -> Scalar { return get_element(t, i, j); }} {}


#ifdef __cpp_concepts
    ElementAccessor(const element_gettable<1> auto& t, std::size_t i, std::invocable auto&& pre_access = []{})
      requires std::same_as<Scalar, scalar_type_of_t<decltype(t)>>
#else
    template<typename T, typename PreAccess = std::function<void()>, std::enable_if_t<
      std::is_constructible_v<std::function<void()>, PreAccess&&> and
      std::is_same_v<Scalar, typename scalar_type_of<T>::type> and element_gettable<const T&, 1>, int> = 0>
    ElementAccessor(const T& t, std::size_t i, PreAccess&& pre_access = []{})
#endif
      : before_access {std::forward<decltype(pre_access)>(pre_access)},
        getter {[&t, i] () -> Scalar { return get_element(t, i); }} {}


    /// Get an element.
    operator Scalar() const
    {
      before_access();
      return getter();
    }

  private:

    const std::function<void()> before_access;

    const std::function<Scalar()> getter;

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<typename T, std::integral I1, std::integral I2, std::invocable PreAccess, std::invocable PostSet>
    requires element_settable<T&, 2> and element_gettable<T&, 2> and (not std::is_const_v<T>)
#else
  template<typename T, typename I1, typename I2, typename PreAccess, typename PostSet, std::enable_if_t<
    element_settable<T&, 2> and element_gettable<T&, 2> and std::is_integral_v<I1> and std::is_integral_v<I2> and
    std::is_invocable_v<PreAccess> and std::is_invocable_v<PostSet> and (not std::is_const_v<T>), int> = 0>
#endif
  ElementAccessor(T&, const I1&, const I2&, PreAccess&&, PostSet&&)
    -> ElementAccessor<true, scalar_type_of_t<T>>;


#ifdef __cpp_concepts
  template<typename  T, std::integral I, std::invocable PreAccess, std::invocable PostSet>
    requires element_settable<T&, 1> and element_gettable<T&, 1> and (not std::is_const_v<T>)
#else
  template<typename T, typename I, typename PreAccess, typename PostSet, std::enable_if_t<
    element_settable<T&, 1> and element_gettable<T&, 1> and std::is_integral_v<I> and
    std::is_invocable_v<PreAccess> and std::is_invocable_v<PostSet> and (not std::is_const_v<T>), int> = 0>
#endif
  ElementAccessor(T&, const I&, PreAccess&&, PostSet&&)
    -> ElementAccessor<true, scalar_type_of_t<T>>;


#ifdef __cpp_concepts
  template<typename T, std::integral I1, std::integral I2, std::invocable PreAccess> requires element_gettable<T&&, 2>
#else
  template<typename T, typename I1, typename I2, typename PreAccess, std::enable_if_t<element_gettable<T&&, 2> and
    std::is_integral_v<I1> and std::is_integral_v<I2> and std::is_invocable_v<PreAccess>, int> = 0>
#endif
  ElementAccessor(T&&, const I1&, const I2&, PreAccess&&)
    -> ElementAccessor<element_settable<T, 2> and std::is_same_v<T&&, std::decay_t<T>&>, scalar_type_of_t<T>>;


#ifdef __cpp_concepts
  template<typename T, std::integral I, std::invocable PreAccess> requires element_gettable<T&&, 1>
#else
  template<typename T, typename I, typename PreAccess, std::enable_if_t<element_gettable<T&&, 1> and
    std::is_integral_v<I> and std::is_invocable_v<PreAccess>, int> = 0>
#endif
  ElementAccessor(T&&, const I&, PreAccess&&)
    -> ElementAccessor<element_settable<T, 1> and std::is_same_v<T&&, std::decay_t<T>&>, scalar_type_of_t<T>>;


#ifdef __cpp_concepts
  template<typename T, std::integral I1, std::integral I2> requires element_gettable<T&&, 2>
#else
  template<typename T, typename I1, typename I2, std::enable_if_t<
    element_gettable<T&&, 2> and std::is_integral_v<I1> and std::is_integral_v<I2>, int> = 0>
#endif
  ElementAccessor(T&&, const I1&, const I2&)
    -> ElementAccessor<element_settable<T, 2> and std::is_same_v<T&&, std::decay_t<T>&>, scalar_type_of_t<T>>;


#ifdef __cpp_concepts
  template<typename T, std::integral I> requires element_gettable<T&&, 1>
#else
  template<typename T, typename I, std::enable_if_t<element_gettable<T&&, 1> and std::is_integral_v<I>, int> = 0>
#endif
  ElementAccessor(T&&, const I&)
    -> ElementAccessor<element_settable<T, 1> and std::is_same_v<T&&, std::decay_t<T>&>, scalar_type_of_t<T>>;


}

#endif //OPENKALMAN_ELEMENTACCESSOR_HPP
