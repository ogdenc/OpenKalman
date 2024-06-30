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
 * \brief Definition for \ref constant_diagonal_coefficient.
 */

#ifndef OPENKALMAN_CONSTANT_DIAGONAL_COEFFICIENT_HPP
#define OPENKALMAN_CONSTANT_DIAGONAL_COEFFICIENT_HPP

#include <type_traits>

namespace OpenKalman::values
{
  /**
   * \brief The constant associated with T, assuming T is a \ref constant_diagonal_matrix.
   * \details If T is a \ref constant_diagonal_matrix, it will have a <code>value</code> defined.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct constant_diagonal_coefficient
  {
    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};
  };


  /**
   * \brief Deduction guide for \ref constant_diagonal_coefficient.
   */
  template<typename T>
  explicit constant_diagonal_coefficient(T&&) -> constant_diagonal_coefficient<std::decay_t<T>>;


  /// Helper template for constant_diagonal_coefficient.
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  constexpr auto constant_diagonal_coefficient_v = constant_diagonal_coefficient<T>::value;


  namespace detail
  {
    template<typename T>
    constexpr auto const_value =
      std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>()))>::value;


#ifdef __cpp_concepts
    template<typename T>
    concept has_static_constant_diagonal =
      interface::get_constant_diagonal_defined_for<T, ConstantType::static_constant> or
      (interface::get_constant_defined_for<T, ConstantType::static_constant> and
        (one_dimensional<T> or internal::are_within_tolerance(const_value<T>, 0)));
#else
    template<typename T, typename = void>
    struct has_static_constant_diagonal_impl : std::false_type {};

    template<typename T>
    struct has_static_constant_diagonal_impl<T, std::enable_if_t<interface::get_constant_defined_for<T, ConstantType::static_constant>>>
      : std::bool_constant<one_dimensional<T> or internal::are_within_tolerance(const_value<T>, 0)> {};


    template<typename T>
    constexpr bool has_static_constant_diagonal =
      interface::get_constant_diagonal_defined_for<T, ConstantType::static_constant> or
      has_static_constant_diagonal_impl<T>::value;
#endif
  } // namespace detail


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::has_static_constant_diagonal<T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<indexible<T> and detail::has_static_constant_diagonal<T>>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

  public:

    constexpr constant_diagonal_coefficient() = default;

    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_diagonal_coefficient;

    static constexpr value_type value = []{
      if constexpr (interface::get_constant_diagonal_defined_for<T>)
        return std::decay_t<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
      else
        return std::decay_t<decltype(Trait::get_constant(std::declval<T>()))>::value;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }
  };


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant can be known only at runtime.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not detail::has_static_constant_diagonal<T>) and
    (interface::get_constant_diagonal_defined_for<T, ConstantType::dynamic_constant> or one_dimensional<T>)
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<indexible<T> and (not detail::has_static_constant_diagonal<T>) and
    (interface::get_constant_diagonal_defined_for<T, ConstantType::dynamic_constant> or one_dimensional<T>)>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

  public:

    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>& t) : m_value {[](const auto& t){
        if constexpr (interface::get_constant_diagonal_defined_for<T>)
          return get_scalar_constant_value(Trait::get_constant_diagonal(t));
        else if constexpr (interface::get_constant_defined_for<T>)
          return get_scalar_constant_value(Trait::get_constant(t));
        else
          return internal::get_singular_component(t);
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_diagonal_coefficient;

    constexpr operator value_type() const noexcept { return m_value; }

    constexpr value_type operator()() const noexcept { return m_value; }

  private:

    value_type m_value;
  };


} // namespace OpenKalman::values


namespace OpenKalman
{
  using values::constant_diagonal_coefficient;
  using values::constant_diagonal_coefficient_v;
}


#endif //OPENKALMAN_CONSTANT_DIAGONAL_COEFFICIENT_HPP
