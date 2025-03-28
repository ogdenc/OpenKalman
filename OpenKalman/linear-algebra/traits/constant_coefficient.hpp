/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref constant_coefficient.
 */

#ifndef OPENKALMAN_CONSTANT_COEFFICIENT_HPP
#define OPENKALMAN_CONSTANT_COEFFICIENT_HPP

#include <type_traits>
#include "values/values.hpp"
#include "linear-algebra/interfaces/object-traits-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/one_dimensional.hpp"

namespace OpenKalman::value
{
  /**
   * \brief The constant associated with T, assuming T is a \ref constant_matrix.
   * \details If T is a \ref constant_matrix, it will have a <code>value</code> defined.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct constant_coefficient
  {
    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};
  };


  /**
   * \brief Deduction guide for \ref constant_coefficient.
   */
  template<typename T>
  explicit constant_coefficient(const T&) -> constant_coefficient<T>;


  /**
   * \brief Helper template for constant_coefficient.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  constexpr auto constant_coefficient_v = constant_coefficient<T>::value;



  namespace detail
  {
    template<typename T>
    constexpr auto const_diag_value =
      std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>()))>::value;


#ifdef __cpp_concepts
    template<typename T>
    concept has_static_constant =
      value::fixed<typename interface::get_constant_return_type<T>::type> or
      (value::fixed<typename interface::get_constant_diagonal_return_type<T>::type> and
        (one_dimensional<T> or value::internal::near(const_diag_value<T>, 0)));
#else
    template<typename T, typename = void>
    struct has_static_constant_impl : std::false_type {};

    template<typename T>
    struct has_static_constant_impl<T, std::enable_if_t<value::fixed<typename interface::get_constant_diagonal_return_type<T>::type>>>
      : std::bool_constant<one_dimensional<T> or value::internal::near(const_diag_value<T>, 0)> {};

    template<typename T>
    constexpr bool has_static_constant = value::fixed<typename interface::get_constant_return_type<T>::type> or
      has_static_constant_impl<T>::value;
#endif
  } // namespace detail


  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::has_static_constant<T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<indexible<T> and detail::has_static_constant<T>>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

  public:

    constexpr constant_coefficient() = default;

    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    static constexpr value_type value = []{
      if constexpr (value::fixed<typename interface::get_constant_return_type<T>::type>)
        return std::decay_t<decltype(Trait::get_constant(std::declval<T>()))>::value;
      else
        return std::decay_t<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
    }();

    constexpr operator value_type() const { return value; }

    constexpr value_type operator()() const { return value; }

  };


  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is unknown at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not detail::has_static_constant<T>) and
    (value::dynamic<typename interface::get_constant_return_type<T>::type> or one_dimensional<T>)
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<
    (value::dynamic<typename interface::get_constant_return_type<T>::type> or one_dimensional<T>) and
    (not detail::has_static_constant<T>)>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

  public:

    explicit constexpr constant_coefficient(const std::decay_t<T>& t) : m_value {[](const auto& t){
        if constexpr (value::dynamic<typename interface::get_constant_return_type<T>::type>)
          return value::to_number(Trait::get_constant(t));
        else if constexpr (value::dynamic<typename interface::get_constant_diagonal_return_type<T>::type>)
          return value::to_number(Trait::get_constant_diagonal(t));
        else
          return internal::get_singular_component(t);
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    constexpr operator value_type() const { return m_value; }

    constexpr value_type operator()() const { return m_value; }

  private:

    value_type m_value;
  };


} // namespace OpenKalman::value


namespace OpenKalman
{
  using value::constant_coefficient;
  using value::constant_coefficient_v;
}

#endif //OPENKALMAN_CONSTANT_COEFFICIENT_HPP
