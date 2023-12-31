/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
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

namespace OpenKalman
{

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
    concept known_constant_diagonal = interface::get_constant_diagonal_defined_for<T, ConstantType::static_constant> or
      (interface::get_constant_defined_for<T, ConstantType::static_constant> and
        (one_dimensional<T, Qualification::depends_on_dynamic_shape> or (square_shaped<T, Qualification::depends_on_dynamic_shape> and
          requires(T t) {
            requires internal::are_within_tolerance(std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant(t))>::value, 0);
          })));
#else
    template<typename T, typename = void>
    struct known_constant_diagonal_impl : std::false_type {};

    template<typename T>
    struct known_constant_diagonal_impl<T, std::enable_if_t<interface::get_constant_defined_for<T, ConstantType::static_constant>>>
      : std::bool_constant<interface::get_constant_defined_for<T, ConstantType::static_constant> and
        (one_dimensional<T, Qualification::depends_on_dynamic_shape> or
          (square_shaped<T, Qualification::depends_on_dynamic_shape> and internal::are_within_tolerance(
            std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant(std::declval<T>()))>::value, 0)))> {};


    template<typename T>
    constexpr bool known_constant_diagonal = interface::get_constant_diagonal_defined_for<T, ConstantType::static_constant> or
      detail::known_constant_diagonal_impl<T>::value;
#endif
  } // namespace detail


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::known_constant_diagonal<T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<indexible<T> and detail::known_constant_diagonal<T>>>
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

    static constexpr Qualification status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Qualification::unqualified;
      else if constexpr (interface::get_constant_diagonal_defined_for<T>)
        return detail::constant_status<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
      else
        return Qualification::depends_on_dynamic_shape;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

  };


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant can be known only at runtime.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not detail::known_constant_diagonal<T>) and
    (interface::get_constant_diagonal_defined_for<T, ConstantType::dynamic_constant> or one_dimensional<T, Qualification::depends_on_dynamic_shape>)
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<indexible<T> and (not detail::known_constant_diagonal<T>) and
    (interface::get_constant_diagonal_defined_for<T, ConstantType::dynamic_constant> or one_dimensional<T, Qualification::depends_on_dynamic_shape>)>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

    template<typename Arg, std::size_t...Ix>
    static constexpr auto get_zero_component(const Arg& arg, std::index_sequence<Ix...>) { return get_component(arg, static_cast<decltype(Ix)>(0)...); }

  public:

    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>& t) : value {[](const auto& t){
        if constexpr (interface::get_constant_diagonal_defined_for<T>)
          return get_scalar_constant_value(Trait::get_constant_diagonal(t));
        else if constexpr (interface::get_constant_defined_for<T>)
          return get_scalar_constant_value(Trait::get_constant(t));
        else
          return get_zero_component(t, std::make_index_sequence<index_count_v<T>>{});
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_diagonal_coefficient;

    static constexpr Qualification status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Qualification::unqualified;
      else if constexpr (interface::get_constant_diagonal_defined_for<T>)
        return detail::constant_status<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
      else
        return Qualification::depends_on_dynamic_shape;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

  private:
    value_type value;
  };


  namespace internal
  {
    template<typename T>
    struct participates_in_constant_arithmetic<constant_diagonal_coefficient<T>> : std::true_type {};
  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_DIAGONAL_COEFFICIENT_HPP
