/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Functions for index descriptors.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP

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
   * \brief Three-way comparison for a non-built-in \ref index_descriptor.
   * \details A comparison of A and B is a partial ordering based on whether or not A is a prefix of B.
   */
  template<index_descriptor A, index_descriptor B>
  constexpr auto operator<=>(const A& a, const B& b) requires (not std::integral<A>) or (not std::integral<B>)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
    {
      if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
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
    else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
    {
      return get_dimension_size_of(a) <=> get_dimension_size_of(b);
    }
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
    {
      if (a.partial_match(b)) return std::partial_ordering {get_dimension_size_of(a) <=> get_dimension_size_of(b)};
      else return std::partial_ordering::unordered;
    }
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
    {
      if (b.partial_match(a)) return std::partial_ordering {get_dimension_size_of(a) <=> get_dimension_size_of(b)};
      else return std::partial_ordering::unordered;
    }
    else
    {
      return std::partial_ordering::unordered;
    }
  }


  /**
   * \brief Equality comparison for non-built-in \ref index_descriptors.
   */
  constexpr bool operator==(const index_descriptor auto& a, const index_descriptor auto& b)
    requires (not std::integral<decltype(a)>) or (not std::integral<decltype(b)>)
  {
    return std::is_eq(a <=> b);
  }
#else
  /**
   * \brief Equivalence comparison for a non-built-in \ref index_descriptor.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
      (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
      return equivalent_to<A, B>;
    else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
      return get_dimension_size_of(a) == get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partial_match(b) and get_dimension_size_of(a) == get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partial_match(a) and get_dimension_size_of(a) == get_dimension_size_of(b);
    else
      return false;
  }


  /**
   * \brief Compares index descriptors for non-equivalence.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }


  /**
   * \brief Determine whether one index descriptor is less than another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
      return prefix_of<A, B>;
    else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
      return get_dimension_size_of(a) < get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partial_match(b) and get_dimension_size_of(a) < get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partial_match(a) and get_dimension_size_of(a) < get_dimension_size_of(b);
    else
      return false;
  }


  /**
   * \brief Determine whether one index descriptor is greater than another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
      return prefix_of<B, A>;
    else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
      return get_dimension_size_of(a) > get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partial_match(b) and get_dimension_size_of(a) > get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partial_match(a) and get_dimension_size_of(a) > get_dimension_size_of(b);
    else
      return false;
  }


  /**
   * \brief Determine whether one index descriptor is less than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return operator<(a, b) or operator==(a, b);
  }


  /**
   * \brief Determine whether one index descriptor is greater than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    return operator>(a, b) or operator==(a, b);
  }
#endif


  // -------------- //
  //   Arithmetic   //
  // -------------- //

  /**
   * \brief Add two \ref index_descriptor values, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<index_descriptor T, index_descriptor U>
#else
  template<typename T, typename U, std::enable_if_t<index_descriptor<T> and index_descriptor<U>, int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (fixed_index_descriptor<T> and fixed_index_descriptor<U>)
    {
      return concatenate_fixed_index_descriptor_t<T, U> {};
    }
    else if constexpr (euclidean_index_descriptor<T> and euclidean_index_descriptor<U>)
    {
      if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
        return Dimensions{get_dimension_size_of(t) + get_dimension_size_of(u)};
      else
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
    }
    else
    {
      return DynamicTypedIndex {std::forward<T>(t), std::forward<U>(u)};
    }
  }


  /**
   * \brief Subtract two \ref euclidean_index_descriptor values, whether fixed or dynamic.
   * \warning This does not perform any runtime checks to ensure that the result is non-negative.
   */
#ifdef __cpp_concepts
  template<euclidean_index_descriptor T, euclidean_index_descriptor U> requires (dimension_size_of_v<T> == dynamic_size) or
    (dimension_size_of_v<U> == dynamic_size) or (dimension_size_of_v<T> > dimension_size_of_v<U>)
#else
  template<typename T, typename U, std::enable_if_t<euclidean_index_descriptor<T> and euclidean_index_descriptor<U> and
    ((dimension_size_of<T>::value == dynamic_size) or (dimension_size_of<U>::value == dynamic_size) or
      (dimension_size_of<T>::value > dimension_size_of<U>::value)), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u) noexcept
  {
    if constexpr (dynamic_index_descriptor<T> or dynamic_index_descriptor<U>)
      return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
    else
      return Dimensions<dimension_size_of_v<T> - dimension_size_of_v<U>>{};
  }


  namespace internal
  {

    // ------------------------------ //
    //   replicate_index_descriptor   //
    // ------------------------------ //

    /**
     * \brief Replicate \ref fixed_index_descriptor T some number of times.
     */
#ifdef __cpp_concepts
    template<floating_scalar_type..., fixed_index_descriptor T, static_index_value N>
#else
    template<typename...S, typename T, typename N, std::enable_if_t<(floating_scalar_type<S> and ...) and
      fixed_index_descriptor<T> and static_index_value<N>, int> = 0>
#endif
    auto replicate_index_descriptor(const T& t, N)
    {
      return replicate_fixed_index_descriptor_t<T, static_index_value_of_v<N>> {};
    }


    /**
     * \overload
     * \brief Replicate a (potentially) dynamic \ref index_descriptor T some (potentially) dynamic number of times.
     */
#ifdef __cpp_concepts
    template<floating_scalar_type...AllowableScalarTypes, index_descriptor T, index_value N>
    requires (not fixed_index_descriptor<T>) or (not static_index_value<N>)
#else
    template<typename...AllowableScalarTypes, typename T, typename N, std::enable_if_t<
      (floating_scalar_type<AllowableScalarTypes> and ...) and index_descriptor<T> and index_value<N> and
        (not fixed_index_descriptor<T> or not static_index_value<N>), int> = 0>
#endif
    auto replicate_index_descriptor(const T& t, N n)
    {
      auto ret = [](const T& t){
        if constexpr (sizeof...(AllowableScalarTypes) > 0) return DynamicTypedIndex<AllowableScalarTypes...> {t};
        else return DynamicTypedIndex {t};
      }(t);
      for (std::size_t i = 1; i < static_cast<std::size_t>(n); ++i) ret.extend(t);
      return ret;
    }


    // --------------------------- //
    //   is_uniform_component_of   //
    // --------------------------- //

    /**
     * \internal
     * \brief Whether a is a 1D index descriptor that, when replicated some number of times, becomes c.
     */
#ifdef __cpp_concepts
    template<index_descriptor A, index_descriptor C>
#else
    template<typename A, typename C, std::enable_if_t<index_descriptor<A> and index_descriptor<C>, int> = 0>
#endif
    constexpr bool is_uniform_component_of(const A& a, const C& c)
    {
      if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<C>)
        return equivalent_to_uniform_dimension_type_of<A, C>;
      else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<C>)
        return get_dimension_size_of(a) == 1;
      else
        return false;
    }


    /**
     * \internal
     * \overload
     */
#ifdef __cpp_concepts
    template<index_descriptor A, typename...S>
#else
    template<typename A, typename...S, std::enable_if_t<index_descriptor<A>, int> = 0>
#endif
    constexpr bool is_uniform_component_of(const A& a, const DynamicTypedIndex<S...>& c)
    {
      if (get_dimension_size_of(a) != 1) return false;
      else if (get_index_descriptor_is_euclidean(a) and get_index_descriptor_is_euclidean(c)) return true;
      else return replicate_index_descriptor<S...>(a, get_dimension_size_of(c)) == c;
    }


    /**
     * \internal
     * \overload
     */
#ifdef __cpp_concepts
    template<typename...T, index_descriptor C>
#else
    template<typename...T, typename C, std::enable_if_t<index_descriptor<C>, int> = 0>
#endif
    constexpr bool is_uniform_component_of(const DynamicTypedIndex<T...>& a, const C& c)
    {
      if (get_dimension_size_of(a) != 1) return false;
      else if (get_index_descriptor_is_euclidean(a) and get_index_descriptor_is_euclidean(c)) return true;
      else return replicate_index_descriptor(a, get_dimension_size_of(c)) == c;
    }


    /**
     * \internal
     * \overload
     */
    template<typename...T, typename...S>
    constexpr bool is_uniform_component_of(const DynamicTypedIndex<T...>& a, const DynamicTypedIndex<S...>& c)
    {
      if constexpr (((not std::is_same_v<T, S>) or ...)) return false;
      else if (get_dimension_size_of(a) != 1) return false;
      else if (get_index_descriptor_is_euclidean(a) and get_index_descriptor_is_euclidean(c)) return true;
      else return replicate_index_descriptor(a, get_dimension_size_of(c)) == c;
    }

  } // namespace internal


} // namespace OpenKalman


#endif //OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
