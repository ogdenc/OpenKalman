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
 * \brief Functions for accessing elements of typed arrays, based on typed coefficients.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP

#include <type_traits>
#include <functional>

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif

namespace OpenKalman
{
  // ------------------------- //
  //   get_dimension_size_of   //
  // ------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_dimension_size_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return dimension_size_of_v<T>;
    else return dimension_size_of<T>::get(t);
  }


  // ----------------------------------- //
  //   get_euclidean_dimension_size_of   //
  // ----------------------------------- //

  /**
   * \brief Get the Euclidean dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_euclidean_dimension_size_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return euclidean_dimension_size_of_v<T>;
    else return euclidean_dimension_size_of<T>::get(t);
  }


  // ------------------------------------------- //
  //   get_index_descriptor_component_count_of   //
  // ------------------------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_index_descriptor_component_count_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return index_descriptor_components_of_v<T>;
    else return index_descriptor_components_of<T>::get(t);
  }


  // ------------------- //
  //  get_dimensions_of  //
  // ------------------- //

#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible Arg> requires (N < max_indices_of_v<Arg>)
#else
  template<std::size_t N = 0, typename Arg, std::enable_if_t<indexible<Arg> and N < max_indices_of<Arg>::value, int> = 0>
#endif
  constexpr auto get_dimensions_of(const Arg& arg)
  {
    using T = coefficient_types_of_t<Arg, N>;
    if constexpr (typed_index_descriptor<T>)
    {
      if constexpr (dynamic_dimension<Arg, N>)
        return interface::CoordinateSystemTraits<std::decay_t<Arg>, N>::coordinate_system_types_at_runtime(std::forward<Arg>(arg));
      else
        return coefficient_types_of_t<Arg, N> {};
    }
    else
    {
      if constexpr (dynamic_dimension<Arg, N>)
        return Dimensions{interface::IndexTraits<std::decay_t<Arg>, N>::dimension_at_runtime(arg)};
      else
        return Dimensions<index_dimension_of_v<Arg, N>> {};
    }
  }


  // --------------------- //
  //  get_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<std::size_t...I, typename T>
    constexpr auto get_tensor_order_of_impl(std::index_sequence<I...>, const T& t)
    {
      return ((get_dimensions_of<I>(t) == 1 ? 0 : 1) + ... + 0);
    }
  }


  /**
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto get_tensor_order_of(const T& t)
  {
    if constexpr (not has_dynamic_dimensions<T>)
      return tensor_order_of_v<T>;
    else
      return detail::get_tensor_order_of_impl(std::make_index_sequence<max_indices_of_v<T>>{}, t);
  }


  // ----------------------- //
  //  get_all_dimensions_of  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>, const T& t)
    {
      return std::tuple {[](const T& t){
        constexpr std::size_t size = index_dimension_of_v<T, I>;
        if constexpr (size == dynamic_size)
          return Dimensions<size>{get_dimensions_of<I>(t)};
        else
          return Dimensions<size>{};
      }(t)...};
    }


    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>)
    {
      return std::tuple {Dimensions<index_dimension_of_v<T, I>>{}...};
    }
  }


  /**
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto) get_all_dimensions_of(T&& t)
  {
    if constexpr (Eigen3::eigen_zero_expr<T> or Eigen3::eigen_constant_expr<T>)
      return std::forward<T>(t).get_all_dimensions();
    else
      return detail::get_all_dimensions_of_impl(std::make_index_sequence<max_indices_of_v<T>>{}, t);
  }


  /**
   * \overload
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \details This overload is only enabled if all dimensions of T are known at compile time.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not has_dynamic_dimensions<T>)
#else
  template<typename T, std::enable_if_t<indexible<T> and not has_dynamic_dimensions<T>, int> = 0>
#endif
  constexpr auto get_all_dimensions_of()
  {
    return detail::get_all_dimensions_of_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{});
  }


  // -------------- //
  //   Comparison   //
  // -------------- //

#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  /**
   * \brief Three-way comparison for a non-built-in \ref index_descriptor.
   */
  template<index_descriptor A, index_descriptor B> requires (not std::integral<A>) and (not std::integral<B>)
  constexpr auto operator<=>(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
    {
      if constexpr (dimension_size_of_v<A> == dimension_size_of_v<B>)
      {
        if constexpr (equivalent_to<A, B>) return std::partial_ordering::equivalent;
        else return std::partial_ordering::unordered;
      }
      else return std::partial_ordering {dimension_size_of_v<A> <=> dimension_size_of_v<B>};
    }
    else if constexpr (untyped_index_descriptor<A> and untyped_index_descriptor<B>)
    {
      return get_dimension_size_of(a) <=> get_dimension_size_of(b);
    }
    else // At least one of A or B is dynamic (DynamicCoefficients or Dimensions).
    {
      auto size_a = get_dimension_size_of(a);
      auto size_b = get_dimension_size_of(b);

      if (size_a == size_b)
      {
        if constexpr (dynamic_index_descriptor<A> and not untyped_index_descriptor<A>)
        {
          if (a.is_equivalent(b)) return std::partial_ordering::equivalent;
          else return std::partial_ordering::unordered;
        }
        else if constexpr (dynamic_index_descriptor<B> and not untyped_index_descriptor<B>)
        {
          if (b.is_equivalent(a)) return std::partial_ordering::equivalent;
          else return std::partial_ordering::unordered;
        }
        else
        {
          return std::partial_ordering::unordered;
        }
      }
      else
      {
        return std::partial_ordering {size_a <=> size_b};
      }
    }
  }


  /**
   * \brief Equality comparison for non-built-in \ref index_descriptors.
   */
  template<index_descriptor A, index_descriptor B> requires (not std::integral<A>) and (not std::integral<B>)
  constexpr bool operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }
#else
  /**
   * \brief Equivalence comparison for a non-built-in \ref index_descriptor.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
      (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
    {
      return equivalent_to<A, B>;
    }
    else if constexpr (untyped_index_descriptor<A> and untyped_index_descriptor<B>)
    {
      return get_dimension_size_of(a) == get_dimension_size_of(b);
    }
    else // At least one of A or B is dynamic (DynamicCoefficients or Dimensions).
    {
      if (get_dimension_size_of(a) == get_dimension_size_of(b))
      {
        if constexpr (dynamic_index_descriptor<A> and not untyped_index_descriptor<A>)
          return a.is_equivalent(b);
        else if constexpr (dynamic_index_descriptor<B> and not untyped_index_descriptor<B>)
          return b.is_equivalent(a);
        else
          return false;
      }
      else
      {
        return false;
      }
    }
  }


  /**
   * \brief Compares index descriptors for non-equivalence.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }


  /**
   * \brief Determine whether one index descriptor is less than another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    return get_dimension_size_of(a) < get_dimension_size_of(b);
  }


  /**
   * \brief Determine whether one index descriptor is greater than another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    return get_dimension_size_of(a) > get_dimension_size_of(b);
  }


  /**
   * \brief Determine whether one index descriptor is less than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return operator<(a, b) or operator==(a, b);
  }


  /**
   * \brief Determine whether one index descriptor is greater than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
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
  template<index_descriptor T, index_descriptor U> requires
    (not (typed_index_descriptor<T> or typed_index_descriptor<U>) or (fixed_index_descriptor<T> and fixed_index_descriptor<U>))
#else
  template<typename T, typename U, std::enable_if_t<index_descriptor<T> and index_descriptor<U> and
    (not (typed_index_descriptor<T> or typed_index_descriptor<U>) or (fixed_index_descriptor<T> and fixed_index_descriptor<U>)), int> = 0>
#endif
  constexpr auto operator+(const T& t, const U& u) noexcept
  {
    if constexpr (typed_index_descriptor<T> or typed_index_descriptor<U>)
    {
      return Concatenate<T, U> {};
    }
    else
    {
      if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
        return Dimensions{get_dimension_size_of(t) + get_dimension_size_of(u)};
      else
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
    }
  }


  /**
   * \brief Subtract two \ref untyped_index_descriptor values, whether fixed or dynamic.
   * \warning This does not perform any runtime checks to ensure that the result is non-negative.
   */
#ifdef __cpp_concepts
  template<untyped_index_descriptor T, untyped_index_descriptor U> requires (dimension_size_of_v<T> == dynamic_size) or
    (dimension_size_of_v<U> == dynamic_size) or (dimension_size_of_v<T> > dimension_size_of_v<U>)
#else
  template<typename T, typename U, std::enable_if_t<untyped_index_descriptor<T> and untyped_index_descriptor<U> and
    ((dimension_size_of<T>::value == dynamic_size) or (dimension_size_of<U>::value == dynamic_size) or
      (dimension_size_of<T>::value > dimension_size_of<U>::value)), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u) noexcept
  {
    if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
      return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
    else
      return Dimensions<dimension_size_of_v<T> - dimension_size_of_v<U>>{};
  }


  namespace internal
  {

    // ----------------------- //
    //  make_dimensions_tuple  //
    // ----------------------- //

    namespace detail
    {
      template<typename T, std::size_t I_begin, std::size_t...Is>
      constexpr auto iterate_dimensions_tuple(std::index_sequence<Is...>)
      {
        return std::tuple {Dimensions<index_dimension_of_v<T, I_begin + Is>>{}...};
      }


      template<typename T, std::size_t I, std::size_t Max>
      constexpr auto make_dimensions_tuple_impl()
      {
        if constexpr (I >= Max)
          return std::tuple{};
        else
          return iterate_dimensions_tuple<T, I>(std::make_index_sequence<Max - I>{});
      }


      template<typename T, std::size_t I, std::size_t Max, typename N, typename...Ns>
      constexpr auto make_dimensions_tuple_impl(N n, Ns...ns)
      {
        static_assert(I < Max);
        if constexpr (dynamic_dimension<T, I>)
          return std::tuple_cat(std::tuple {Dimensions{n}}, make_dimensions_tuple_impl<T, I + 1, Max>(ns...));
        else
          return std::tuple_cat(std::tuple {Dimensions<index_dimension_of_v<T, I>>{}},
            make_dimensions_tuple_impl<T, I + 1, Max>(n, ns...));
      }
    }


#ifdef __cpp_concepts
    template<indexible T, std::convertible_to<const std::size_t> ... N>
    requires (sizeof...(N) == number_of_dynamic_indices_v<T>)
#else
    template<typename T, typename...N, std::enable_if_t<indexible<T> and
      (std::is_convertible_v<N, const std::size_t> and ...) and (sizeof...(N) == number_of_dynamic_indices<T>::value), int> = 0>
#endif
    constexpr auto make_dimensions_tuple(N...n)
    {
      return detail::make_dimensions_tuple_impl<T, 0, max_indices_of_v<T>>(static_cast<const std::size_t>(n)...);
    }

  } // namespace internal

}

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Get a coordinate in Euclidean space corresponding to a coefficient in a matrix with typed coefficients.
   * \details This overload is operable for \ref dynamic_index_descriptor.
   * \tparam T The associated tensor/matrix/array object.
   * \param indices The relevant indices of T.
   * \return The scalar value of the transformed coordinate in Euclidean space corresponding to the provided indices.
   */
#ifdef __cpp_concepts
  template<indexible T, std::convertible_to<const std::size_t>...Indices>
  requires (sizeof...(Indices) == max_indices_of_v<T>)
#else
  template<typename T, typename...Indices, std::enable_if_t<indexible<T> and
    (std::is_convertible_v<Indices, const std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  to_euclidean_element(const T& t, Indices...indices)
  {
    if constexpr (typed_index_descriptor<coefficient_types_of_t<T, 0>>)
      return coefficient_types_of_t<T, 0>::to_euclidean_element(t, 0, 0, indices...);
    else
      return get_dimensions_of<0>(t).to_euclidean_element(t, indices...);
  }


  /**
   * \internal
   * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
   * \details This overload is operable for \ref dynamic_index_descriptor.
   * \tparam T The associated tensor/matrix/array object.
   * \param indices The relevant indices of T.
   * \return The scalar value of the typed coefficient corresponding to the provided indices.
   */
#ifdef __cpp_concepts
  template<indexible T, std::convertible_to<const std::size_t>...Indices>
  requires (sizeof...(Indices) == max_indices_of_v<T>)
#else
  template<typename T, typename...Indices, std::enable_if_t<indexible<T> and
    (std::is_convertible_v<Indices, const std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  from_euclidean_element(const T& t, Indices...indices)
  {
    if constexpr (typed_index_descriptor<coefficient_types_of_t<T, 0>>)
      return coefficient_types_of_t<T, 0>::from_euclidean_element(t, 0, 0, indices...);
    else
      return get_dimensions_of<0>(t).from_euclidean_element(t, indices...);
  }


  /**
   * \internal
   * \brief Wrap a given coefficient and return its wrapped, scalar value.
   * \details This overload is operable for \ref dynamic_index_descriptor.
   * \tparam T The associated tensor/matrix/array object.
   * \param indices The relevant indices of T.
   * \return The scalar value of the wrapped coefficient corresponding to the provided indices.
   */
#ifdef __cpp_concepts
  template<indexible T, std::convertible_to<const std::size_t>...Indices>
  requires (sizeof...(Indices) == max_indices_of_v<T>)
#else
  template<typename T, typename...Indices, std::enable_if_t<indexible<T> and
    (std::is_convertible_v<Indices, const std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  wrap_get_element(const T& t, Indices...indices)
  {
    if constexpr (typed_index_descriptor<coefficient_types_of_t<T, 0>>)
      return coefficient_types_of_t<T, 0>::wrap_get_element(t, 0, indices...);
    else
      return get_dimensions_of<0>(t).wrap_get_element(t, indices...);
  }


  /**
   * \internal
   * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
   * \details This overload is operable for \ref dynamic_index_descriptor.
   * \tparam T The associated tensor/matrix/array object.
   * \param scalar The new element value
   * \param indices The relevant indices of T.
   */
#ifdef __cpp_concepts
  template<indexible T, std::convertible_to<const std::size_t>...Indices>
  requires (sizeof...(Indices) == max_indices_of_v<T>)
#else
  template<typename T, typename...Indices, std::enable_if_t<indexible<T> and
    (std::is_convertible_v<Indices, const std::size_t> and ...), int> = 0>
#endif
  inline void
  wrap_set_element(T& t, const scalar_type_of_t<T> s, Indices...indices)
  {
    if constexpr (typed_index_descriptor<coefficient_types_of_t<T, 0>>)
      coefficient_types_of_t<T, 0>::wrap_set_element(t, s, 0, indices...);
    else
      get_dimensions_of<0>(t).wrap_set_element(t, s, indices...);
  }


}// namespace OpenKalman::internal


#endif //OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
