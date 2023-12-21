/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Comparison operators for \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_COMPARISON_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_COMPARISON_OPERATORS_HPP

#include <type_traits>

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif

namespace OpenKalman
{
  // -------------- //
  //   Comparison   //
  // -------------- //

  namespace internal
  {
    template<typename T>
    struct is_DynamicTypedIndex : std::false_type {};

    template<typename...AllowableScalarTypes>
    struct is_DynamicTypedIndex<DynamicTypedIndex<AllowableScalarTypes...>> : std::true_type {};
  } // namespace detail


#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  /**
   * \brief Three-way comparison for a non-built-in \ref vector_space_descriptor.
   * \details A comparison of A and B is a partial ordering based on whether or not A is a prefix of B.
   */
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr auto operator<=>(const A& a, const B& b) requires (not std::integral<A>) or (not std::integral<B>)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
    {
      if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
        return dimension_size_of_v<A> <=> dimension_size_of_v<B>;
      else if constexpr (equivalent_to<A, B>)
        return std::partial_ordering::equivalent;
      else if constexpr (prefix_of<A, B>)
        return std::partial_ordering::less;
      else if constexpr (prefix_of<B, A>)
        return std::partial_ordering::greater;
      else
        return std::partial_ordering::unordered;
    }
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
    {
      return static_cast<std::size_t>(get_dimension_size_of(a)) <=> static_cast<std::size_t>(get_dimension_size_of(b));
    }
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
    {
      if (a.partially_matches(b))
        return std::partial_ordering {static_cast<std::size_t>(get_dimension_size_of(a)) <=> static_cast<std::size_t>(get_dimension_size_of(b))};
      else
        return std::partial_ordering::unordered;
    }
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
    {
      if (b.partially_matches(a))
        return std::partial_ordering {static_cast<std::size_t>(get_dimension_size_of(a)) <=> static_cast<std::size_t>(get_dimension_size_of(b))};
      else
        return std::partial_ordering::unordered;
    }
    else
    {
      return std::partial_ordering::unordered;
    }
  }


  /**
   * \brief Equality comparison for non-built-in \ref vector_space_descriptor.
   */
  constexpr bool operator==(const vector_space_descriptor auto& a, const vector_space_descriptor auto& b)
    requires (not std::integral<decltype(a)>) or (not std::integral<decltype(b)>)
  {
    return std::is_eq(a <=> b);
  }
#else
  /**
   * \brief Equivalence comparison for a non-built-in \ref vector_space_descriptor.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
      (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return equivalent_to<A, B>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partially_matches(b) and static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partially_matches(a) and static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    else
      return false;
  }


  /**
   * \brief Compares \ref vector_space_descriptor objects for non-equivalence.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is less than another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return prefix_of<A, B>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return static_cast<std::size_t>(get_dimension_size_of(a)) < static_cast<std::size_t>(get_dimension_size_of(b));
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partially_matches(b) and static_cast<std::size_t>(get_dimension_size_of(a)) < static_cast<std::size_t>(get_dimension_size_of(b));
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partially_matches(a) and static_cast<std::size_t>(get_dimension_size_of(a)) < static_cast<std::size_t>(get_dimension_size_of(b));
    else
      return false;
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is greater than another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return prefix_of<B, A>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return static_cast<std::size_t>(get_dimension_size_of(a)) > static_cast<std::size_t>(get_dimension_size_of(b));
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partially_matches(b) and static_cast<std::size_t>(get_dimension_size_of(a)) > static_cast<std::size_t>(get_dimension_size_of(b));
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partially_matches(a) and static_cast<std::size_t>(get_dimension_size_of(a)) > static_cast<std::size_t>(get_dimension_size_of(b));
    else
      return false;
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is less than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return operator<(a, b) or operator==(a, b);
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is greater than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    return operator>(a, b) or operator==(a, b);
  }
#endif


} // namespace OpenKalman


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_COMPARISON_OPERATORS_HPP
