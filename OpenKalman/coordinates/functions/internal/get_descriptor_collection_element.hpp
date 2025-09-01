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
 * \brief Definition for \ref coordinates::internal::get_descriptor_collection_element.
 */

#ifndef OPENKALMAN_COORDINATES_GET_DESCRIPTOR_COLLECTION_ELEMENT_HPP
#define OPENKALMAN_COORDINATES_GET_DESCRIPTOR_COLLECTION_ELEMENT_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/descriptor.hpp"
#include "coordinates/concepts/descriptor_collection.hpp"
#include "coordinates/descriptors/Any.hpp"

namespace OpenKalman::coordinates::internal
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
    template<typename Tup, std::size_t i> requires (i + 1 == collections::size_of_v<Tup>)
    struct common_descriptor_tuple_element<Tup, i>
#else
    template<typename Tup, std::size_t i>
    struct common_descriptor_tuple_element<Tup, i, std::enable_if_t<(i + 1 == collections::size_of_v<Tup>)>>
#endif
    {
      using type = std::decay_t<collections::collection_element_t<i, Tup>>;
    };

#ifdef __cpp_concepts
    template<typename Tup, std::size_t i> requires (i + 1 < collections::size_of_v<Tup>)
    struct common_descriptor_tuple_element<Tup, i>
#else
    template<typename Tup, std::size_t i>
    struct common_descriptor_tuple_element<Tup, i, std::enable_if_t<(i + 1 < collections::size_of_v<Tup>)>>
#endif
    {
      using type = typename common_descriptor_tuple_element_impl<collections::collection_element_t<i, Tup>,
        typename common_descriptor_tuple_element<Tup, i + 1>::type>::type;
    };
  }


  /**
   * \internal
   * \brief Get an element of a \ref coordinates::descriptor_collection
   * \details If components are of different types, the result will be a value of <code>coordinates::Any<double></code>.
   */
#ifdef __cpp_concepts
  template<descriptor_collection Arg, values::index I>
#else
  template<typename Arg, typename I, std::enable_if_t<descriptor_collection<Arg> and values::index<I>, int> = 0>
#endif
  constexpr auto
  get_descriptor_collection_element(Arg&& arg, const I i)
  {
    if constexpr (values::fixed_value_compares_with<collections::size_of<Arg>, 0>)
    {
      return std::integral_constant<std::size_t, 0>{};
    }
    else if constexpr (stdcompat::ranges::random_access_range<Arg>)
    {
      return collections::get(std::forward<Arg>(arg), std::move(i));
    }
    else // collections::size_of_v<Arg> >= 2)
    {
      using Common = typename detail::common_descriptor_tuple_element<std::decay_t<Arg>>::type;
      return collections::get(static_cast<Common>(std::forward<Arg>(arg)), std::move(i));
    }
  };


}

#endif
