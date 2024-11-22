/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_vector_space_descriptor function.
 */

#ifndef OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_HPP


namespace OpenKalman
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct count_is_zero : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires (std::decay_t<decltype(count_indices(std::declval<const T&>()))>::value == 0)
    struct count_is_zero<T>
#else
    template<typename T>
    struct count_is_zero<T, std::enable_if_t<std::decay_t<decltype(count_indices(std::declval<const T&>()))>::value == 0>>
#endif
      : std::true_type {};

  } // namespace detail


  /**
   * \brief Get the \ref vector_space_descriptor object for index N of \ref indexible object T.
   */
#ifdef __cpp_concepts
  template<typename T, value::index N> requires
    interface::get_vector_space_descriptor_defined_for<T> or detail::count_is_zero<T>::value
  constexpr vector_space_descriptor auto
#else
  template<typename T, typename N, std::enable_if_t<value::index<N> and
    (interface::get_vector_space_descriptor_defined_for<T> or detail::count_is_zero<T>::value), int> = 0>
  constexpr auto
#endif
  get_vector_space_descriptor(const T& t, const N& n)
  {
    if constexpr (detail::count_is_zero<T>::value)
    {
      return descriptor::Dimensions<1>{};
    }
    else if constexpr (value::static_index<N> and value::static_index<decltype(count_indices(t))>)
    {
      if constexpr (N::value < std::decay_t<decltype(count_indices(t))>::value)
        return interface::indexible_object_traits<T>::get_vector_space_descriptor(t, n);
      else
        return descriptor::Dimensions<1>{};
    }
    else if constexpr (euclidean_vector_space_descriptor<decltype(interface::indexible_object_traits<T>::get_vector_space_descriptor(t, n))>)
    {
      if (n < count_indices(t))
        return static_cast<std::size_t>(interface::indexible_object_traits<T>::get_vector_space_descriptor(t, n));
      else
        return 1_uz;
    }
    else
    {
      using Scalar = typename interface::indexible_object_traits<std::decay_t<T>>::scalar_type;
      if (n < count_indices(t))
        return descriptor::DynamicDescriptor<Scalar>{interface::indexible_object_traits<T>::get_vector_space_descriptor(t, n)};
      else
        return descriptor::DynamicDescriptor<Scalar>{descriptor::Dimensions<1>{}};
    }
  }


  /**
   * \overload
   * \tparam N An index value known at compile time.
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, typename T> requires
    interface::get_vector_space_descriptor_defined_for<T> or detail::count_is_zero<T>::value
  constexpr vector_space_descriptor auto
#else
  template<std::size_t N = 0, typename T, std::enable_if_t<
    (interface::get_vector_space_descriptor_defined_for<T> or detail::count_is_zero<T>::value), int> = 0>
  constexpr auto
#endif
  get_vector_space_descriptor(const T& t)
  {
    return get_vector_space_descriptor(t, std::integral_constant<std::size_t, N>{});
  }


} // namespace OpenKalman

#endif //OPENKALMAN_GET_VECTOR_SPACE_DESCRIPTOR_HPP
