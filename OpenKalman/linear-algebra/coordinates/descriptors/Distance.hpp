/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
#include <type_traits>
#include "collections/functions/get.hpp"
#include "linear-algebra/coordinates/functions/comparison-operators.hpp"
#include "Any.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \struct Distance
   * \brief A non-negative real or integral number, [0,&infin;], representing a distance.
   * \details This is similar to Axis, but wrapping occurs to ensure that values are never negative.
   */
  struct Distance {};


} // namespace OpenKalman::coordinates


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Distance.
   */
  template<>
  struct coordinate_descriptor_traits<coordinates::Distance>
  {
  private:

    using T = coordinates::Distance;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto dimension = [](const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto stat_dimension = [](const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto is_euclidean = [](const T&) { return std::false_type{}; };


    static constexpr auto hash_code = [](const T&) -> std::size_t
    {
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32) return 0xBD0A_uz;
      else if constexpr (bits < 64) return 0xBD0A6689_uz;
      else return 0xBD0A668977D34578_uz;
    };


    /**
     * \brief This wraps the argument value so that its real part is non-negative.
     */
    static constexpr auto
    to_stat_space = [](const T&, auto&& data_view)
    {
      decltype(auto) d = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, 0>{});
      // The distance component is wrapped to the non-negative half of the real axis:
      return std::array {values::internal::update_real_part(std::forward<decltype(d)>(d), values::abs(values::real(d)))};
    };


    /*
     * \brief This is effectively an identity function.
     * \details This value should be positive, but this performs no bounds checking.
     */
    static constexpr auto
    from_stat_space = [](const T&, auto&& data_view) -> decltype(auto)
    {
      return std::forward<decltype(data_view)>(data_view);
    };


    /*
     * \brief Wraps the argument so that its real part is non-negative.
     */
    static constexpr auto
    wrap = to_stat_space;

  };

}


namespace std
{
  template<>
  struct common_type<OpenKalman::coordinates::Distance, OpenKalman::coordinates::Distance>
  {
    using type = OpenKalman::coordinates::Distance;
  };


  template<typename Scalar>
  struct common_type<OpenKalman::coordinates::Distance, OpenKalman::coordinates::Any<Scalar>>
    : common_type<OpenKalman::coordinates::Any<Scalar>, OpenKalman::coordinates::Distance> {};


  template<typename T>
  struct common_type<OpenKalman::coordinates::Distance, T>
    : std::conditional_t<
      OpenKalman::coordinates::descriptor<T>,
      OpenKalman::stdcompat::type_identity<OpenKalman::coordinates::Any<>>,
      std::monostate> {};
}


#endif //OPENKALMAN_DISTANCE_HPP
