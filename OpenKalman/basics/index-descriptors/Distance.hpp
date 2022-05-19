/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Distance class.
 */

#ifndef OPENKALMAN_DISTANCE_HPP
#define OPENKALMAN_DISTANCE_HPP

#include <cmath>
#include <array>
#include <functional>

namespace OpenKalman
{
  /**
   * \struct Distance
   * \brief A non-negative real or integral number, [0,&infin;], representing a distance.
   * \details This is similar to Axis, but wrapping occurs to ensure that values are never negative.
   */
  struct Distance
  {
    /**
     * \brief Maps an element to positive coordinates in 1D Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
     * \param start The starting index within the index descriptor
     */
#ifdef __cpp_concepts
    template<typename Scalar> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_arithmetic<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    to_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t euclidean_local_index, std::size_t start)
    {
      return g(start);
    }


    /**
     * \brief Maps a coordinate in positive 1D Euclidean space to an element.
     * \details The resulting distance should always be positive, so this function takes the absolute value.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting index within the Euclidean-transformed indices
     */
#ifdef __cpp_concepts
    template<typename Scalar> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_arithmetic<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    from_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t euclidean_start)
    {
      return std::abs(g(euclidean_start)); // The distance component may need to be wrapped to the positive half of the axis.
    }


    /**
     * \brief Perform modular wrapping of an element.
     * \details The wrapping operation is equivalent to taking the absolute value.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<typename Scalar> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_arithmetic<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    wrap_get_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start)
    {
      return std::abs(g(start));
    }


    /**
     * \brief Set an element and then perform any necessary modular wrapping.
     * \details The operation is equivalent to setting and then changing to the absolute value.
     * \tparam Scalar The scalar type (e.g., double).
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<typename Scalar> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_arithmetic<Scalar>::value, int> = 0>
#endif
    static constexpr void
    wrap_set_element(const std::function<void(Scalar, std::size_t)>& s, const std::function<Scalar(std::size_t)>& g,
            Scalar x, std::size_t local_index, std::size_t start)
    {
      s(std::abs(x), start);
    }

  };


  namespace interface
  {
    /**
     * \internal
     * \brief Distance is represented by one coordinate.
     */
    template<>
    struct IndexDescriptorSize<Distance> : std::integral_constant<std::size_t, 1>
    {
      constexpr static std::size_t get(const Distance&) { return 1; }
    };


    /**
     * \internal
     * \brief Distance is represented by one coordinate in Euclidean space.
     */
    template<>
    struct EuclideanIndexDescriptorSize<Distance> : std::integral_constant<std::size_t, 1>
    {
      constexpr static std::size_t get(const Distance&) { return 1; }
    };


    /**
     * \internal
     * \brief The number of atomic components.
     */
    template<>
    struct IndexDescriptorComponentCount<Distance> : std::integral_constant<std::size_t, 1>
    {
      constexpr static std::size_t get(const Distance&) { return 1; }
    };


    /**
     * \internal
     * \brief The type of the result when subtracting two Distance values.
     * \details A difference between two distances can be positive or negative, and is treated as Axis.
     * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
     * 18th Int'l Conf. on Information Fusion 1553, 1555 (2015).
     */
    template<>
    struct IndexDescriptorDifferenceType<Distance> { using type = Axis; };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_DISTANCE_HPP
