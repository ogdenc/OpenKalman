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
 * \brief Definition for \ref constant_coefficient.
 */

#ifndef OPENKALMAN_CONSTANT_COEFFICIENT_HPP
#define OPENKALMAN_CONSTANT_COEFFICIENT_HPP

#include <type_traits>

namespace OpenKalman
{

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
    concept known_constant = interface::get_constant_defined_for<T, ConstantType::static_constant> or
      (interface::get_constant_diagonal_defined_for<T, ConstantType::static_constant> and
        (one_dimensional<T, Qualification::depends_on_dynamic_shape> or requires(T t) {
          requires internal::are_within_tolerance(std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(t))>::value, 0);
        }));
#else
    template<typename T, typename = void>
    struct known_constant_impl : std::false_type {};

    template<typename T>
    struct known_constant_impl<T, std::enable_if_t<interface::get_constant_diagonal_defined_for<T, ConstantType::static_constant>>>
      : std::bool_constant<one_dimensional<T, Qualification::depends_on_dynamic_shape> or internal::are_within_tolerance(
          std::decay_t<decltype(interface::indexible_object_traits<std::decay_t<T>>::get_constant_diagonal(std::declval<T>()))>::value, 0)> {};

    template<typename T>
    constexpr bool known_constant = interface::get_constant_defined_for<T, ConstantType::static_constant> or known_constant_impl<T>::value;
#endif


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct constant_status : std::integral_constant<Qualification, Qualification::unqualified> {};

#ifdef __cpp_concepts
    template<typename T> requires (std::decay_t<T>::status == Qualification::depends_on_dynamic_shape)
    struct constant_status<T>
#else
    template<typename T>
    struct constant_status<T, std::enable_if_t<std::decay_t<T>::status == Qualification::depends_on_dynamic_shape>>
#endif
      : std::integral_constant<Qualification, Qualification::depends_on_dynamic_shape> {};

  } // namespace detail


  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::known_constant<T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<indexible<T> and detail::known_constant<T>>>
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
      if constexpr (interface::get_constant_defined_for<T, ConstantType::static_constant>)
        return std::decay_t<decltype(Trait::get_constant(std::declval<T>()))>::value;
      else
        return std::decay_t<decltype(Trait::get_constant_diagonal(std::declval<T>()))>::value;
    }();

    static constexpr Qualification status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Qualification::unqualified;
      else if constexpr (interface::get_constant_defined_for<T, ConstantType::static_constant>)
        return detail::constant_status<decltype(Trait::get_constant(std::declval<T>()))>::value;
      else
        return Qualification::depends_on_dynamic_shape;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

  };


  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is unknown at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not detail::known_constant<T>) and
    (interface::get_constant_defined_for<T, ConstantType::dynamic_constant> or one_dimensional<T, Qualification::depends_on_dynamic_shape>)
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<(not detail::known_constant<T>) and
    (interface::get_constant_defined_for<T, ConstantType::dynamic_constant> or one_dimensional<T, Qualification::depends_on_dynamic_shape>)>>
#endif
  {
  private:

    using Trait = interface::indexible_object_traits<std::decay_t<T>>;

    template<typename Arg, std::size_t...Ix>
    static constexpr auto get_zero_component(const Arg& arg, std::index_sequence<Ix...>)
    {
      return get_component(arg, static_cast<decltype(Ix)>(0)...);
    }

  public:

    explicit constexpr constant_coefficient(const std::decay_t<T>& t) : value {[](const auto& t){
        if constexpr (interface::get_constant_defined_for<T, ConstantType::dynamic_constant>)
          return get_scalar_constant_value(Trait::get_constant(t));
        else if constexpr (interface::get_constant_diagonal_defined_for<T, ConstantType::dynamic_constant>)
          return get_scalar_constant_value(Trait::get_constant_diagonal(t));
        else
          return get_zero_component(t, std::make_index_sequence<index_count_v<T>>{});
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    static constexpr Qualification status = []{
      if constexpr (not has_dynamic_dimensions<T>)
        return Qualification::unqualified;
      else if constexpr (interface::get_constant_defined_for<T, ConstantType::dynamic_constant>)
        return detail::constant_status<decltype(Trait::get_constant(std::declval<T>()))>::value;
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
    struct participates_in_constant_arithmetic<constant_coefficient<T>> : std::true_type {};
  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_CONSTANT_COEFFICIENT_HPP
