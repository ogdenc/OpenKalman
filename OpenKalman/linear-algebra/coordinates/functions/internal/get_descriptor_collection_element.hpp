/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref coordinate::internal::get_descriptor_collection_element.
 */

#ifndef OPENKALMAN_COORDINATE_GET_DESCRIPTOR_COLLECTION_ELEMENT_HPP
#define OPENKALMAN_COORDINATE_GET_DESCRIPTOR_COLLECTION_ELEMENT_HPP

#include <type_traits>
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_range.hpp"
#include "linear-algebra/coordinates/concepts/descriptor_collection.hpp"

namespace OpenKalman::coordinate
{
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct Any;
}


namespace OpenKalman::coordinate::internal
{
  namespace detail
  {
    template<typename T1, typename T2>
    struct common_descriptor_type { using type = Any<double>; };

    template<typename Scalar1, typename Scalar2>
    struct common_descriptor_type<Any<Scalar1>, Any<Scalar2>> { using type = Any<std::common_type_t<Scalar1, Scalar2>>; };

    template<typename Scalar, typename T>
    struct common_descriptor_type<Any<Scalar>, T> { using type = Any<Scalar>; };

    template<typename T, typename Scalar>
    struct common_descriptor_type<T, Any<Scalar>> { using type = Any<Scalar>; };


#ifdef __cpp_concepts
    template<typename T1, typename T2>
#else
    template<typename T1, typename T2, typename = void>
#endif
    struct common_descriptor_tuple_element_impl : common_descriptor_type<T1, T2> {};

#ifdef __cpp_concepts
    template<typename T1, std::common_with<T1> T2>
    struct common_descriptor_tuple_element_impl<T1, T2>
#else
    template<typename T1, typename T2>
    struct common_descriptor_tuple_element_impl<T1, T2, std::void_t<typename std::common_type<T1, T2>::type>>
#endif
      : std::common_type<T1, T2> {};


#ifdef __cpp_concepts
    template<typename Tup, std::size_t i = 0>
#else
    template<typename Tup, std::size_t i = 0, typename = void>
#endif
    struct common_descriptor_tuple_element;

#ifdef __cpp_concepts
    template<typename Tup, std::size_t i> requires (i + 1 == std::tuple_size_v<Tup>)
    struct common_descriptor_tuple_element<Tup, i>
#else
    template<typename Tup, std::size_t i>
    struct common_descriptor_tuple_element<Tup, i, std::enable_if_t<(i + 1 == std::tuple_size_v<Tup>)>>
#endif
    {
      using type = std::decay_t<std::tuple_element_t<i, Tup>>;
    };

#ifdef __cpp_concepts
    template<typename Tup, std::size_t i> requires (i + 1 < std::tuple_size_v<Tup>)
    struct common_descriptor_tuple_element<Tup, i>
#else
    template<typename Tup, std::size_t i>
    struct common_descriptor_tuple_element<Tup, i, std::enable_if_t<(i + 1 < std::tuple_size_v<Tup>)>>
#endif
    {
      using type = typename common_descriptor_tuple_element_impl<std::tuple_element_t<i, Tup>,
        typename common_descriptor_tuple_element<Tup, i + 1>::type>::type;
    };
  } // namespace detail


  /**
   * \internal
   * \brief Get an element of a \ref coordinate::descriptor_collection
   * \details If components are of different types, the result will be a value of <code>coordinate::Any<double></code>.
   */
#ifdef __cpp_concepts
  template<descriptor_collection Arg, value::index I>
#else
  template<typename Arg, typename I, std::enable_if_t<descriptor_collection<Arg> and value::index<I>, int> = 0>
#endif
  constexpr auto
  get_descriptor_collection_element(Arg&& arg, const I i)
  {
    if constexpr (sized_random_access_range<Arg>)
    {
      return value::internal::get_collection_element(std::forward<Arg>(arg), std::move(i));
    }
    else if constexpr (std::tuple_size_v<std::decay_t<Arg>> == 0)
    {
      return std::integral_constant<std::size_t, 0>{};
    }
    else if constexpr (std::tuple_size_v<std::decay_t<Arg>> == 1)
    {
      return value::internal::get_collection_element(std::forward<Arg>(arg), std::move(i));
    }
    else //if constexpr (descriptor_tuple<Arg> and std::tuple_size_v<std::decay_t<Arg>> >= 2)
    {
      using Common = typename detail::common_descriptor_tuple_element<std::decay_t<Arg>>::type;
      return value::internal::get_collection_element<Common>(std::forward<Arg>(arg), std::move(i));
    }
  };


} // namespace OpenKalman::coordinate::internal

#endif //OPENKALMAN_COORDINATE_GET_DESCRIPTOR_COLLECTION_ELEMENT_HPP
