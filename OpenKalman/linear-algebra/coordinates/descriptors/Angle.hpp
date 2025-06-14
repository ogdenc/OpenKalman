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
 * \brief Definition of the Angle class and related limits.
 */

#ifndef OPENKALMAN_ANGLE_HPP
#define OPENKALMAN_ANGLE_HPP

#include <type_traits>
#include <cmath>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "basics/compatibility/language-features.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/value.hpp"
#include "values/classes/fixed-constants.hpp"
#include "values/functions/internal/update_real_part.hpp"
#include "values/math/sin.hpp"
#include "values/math/cos.hpp"
#include "values/math/atan2.hpp"
#include "values/functions/cast_to.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/traits/common_collection_type.hpp"
#include "collections/views/generate.hpp"
#include "collections/views/update.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [Min,Max) when it increases or decreases outside that range.
   * There are several predefined angles, including angle::Radians, angle::Degrees, angle::PositiveRadians,
   * angle::PositiveDegrees, and angle::Circle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam Min A \ref values::fixed "fixed value" representing the minimum value beyond which wrapping occurs. This must be no greater than 0.
   * \tparam Max A \ref values::fixed "fixed value" representing the maximum value beyond which wrapping occurs. This must be greater than 0.
   */
#ifdef __cpp_concepts
  template<values::fixed Min = values::fixed_minus_pi<long double>, values::fixed Max = values::fixed_pi<long double>>
  requires (values::fixed_number_of_v<Min> <= 0) and (values::fixed_number_of_v<Max> > 0) and
    std::common_with<long double, values::number_type_of_t<Min>> and
    std::common_with<long double, values::number_type_of_t<Max>> and
    std::common_with<values::number_type_of_t<Min>, values::number_type_of_t<Max>>
#else
template<typename Min = values::fixed_minus_pi<long double>, typename Max = values::fixed_pi<long double>>
#endif
  struct Angle
  {
#ifndef __cpp_concepts
    static_assert(values::fixed<Min>);
    static_assert(values::fixed<Max>);
    static_assert(values::fixed_number_of_v<Min> <= 0);
    static_assert(values::fixed_number_of_v<Max> > 0);
#endif
  };


  /// Namespace for definitions relating to specialized instances of \ref Angle.
  namespace angle
  {
    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<>;


    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<values::Fixed<long double, 0>, values::fixed_2pi<long double>>;


    /// An angle measured in degrees [0,360).
    using PositiveDegrees = Angle<values::Fixed<long double, 0>, values::Fixed<long double, 360>>;


    /// An angle measured in positive or negative degrees [-180,180).
    using Degrees = Angle<values::Fixed<long double, -180>, values::Fixed<long double, 180>>;


    /// An wrapping circle such as the wrapping interval [0,1).
    using Circle = Angle<values::Fixed<long double, 0>, values::Fixed<long double, 1>>;


    namespace detail
    {
      template<typename T>
      struct is_angle : std::false_type {};

      template<typename Min, typename Max>
      struct is_angle<Angle<Min, Max>> : std::true_type {};
    }


    /**
     * \brief T is a \ref coordinates::pattern object representing an angle.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept angle =
#else
    static constexpr bool angle =
#endif
      detail::is_angle<T>::value;

  } // namespace angle


} // OpenKalman::coordinates


namespace OpenKalman::interface
{

  /**
   * \internal
   * \brief traits for Angle.
   */
  template<typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Angle<Min, Max>>
  {
  private:

    using T = coordinates::Angle<Min, Max>;
    static constexpr auto min = values::fixed_number_of_v<Min>;
    static constexpr auto max = values::fixed_number_of_v<Max>;


  public:

    static constexpr bool is_specialized = true;


    static constexpr auto dimension = [](const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto stat_dimension = [](const T&) { return std::integral_constant<std::size_t, 2>{}; };


    static constexpr auto is_euclidean = [](const T&) { return std::false_type{}; };


    static constexpr auto hash_code = [](const T&) -> std::size_t
    {

      constexpr auto min_float = static_cast<float>(min);
      constexpr auto max_float = static_cast<float>(max);
      constexpr float a = (min_float * 3.f + max_float * 2.f) / (max_float - min_float);
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32) return 0x62BB_uz + static_cast<std::size_t>(a * a * 0x1.p2f);
      else if constexpr (bits < 64) return 0x62BB0D37_uz + static_cast<std::size_t>(a * a * 0x1.p4f);
      else return 0x62BB0D37A58D6F96_uz + static_cast<std::size_t>(a * a * 0x1.p8f);
    };

  private:

    template<typename Scalar>
    struct to_stat_collection
    {
      to_stat_collection() = default;
      explicit constexpr to_stat_collection(Scalar theta) : my_theta {std::move(theta)} {};
      constexpr auto operator()(std::size_t i) const { return i == 0 ? values::cos(my_theta) : values::sin(my_theta); };
    private:
      Scalar my_theta;
    };

  public:

    /*
     * \details Maps the angle to corresponding x and y coordinates on a unit circle.
     * By convention, the minimum angle limit Limits<Scalar::min corresponds to the point (-1,0) in Euclidean statistical space,
     * and the angle is scaled so that the difference between Limits<Scalar>::min and Limits<<Scalar>::max is 2&pi;,
     * so Limits<Scalar>::max wraps back to the point (-1, 0).
     */
    static constexpr auto
    to_stat_space =
#ifdef __cpp_concepts
    [](const T&, const collections::collection_view auto& data_view) noexcept
#else
    [](const T&, const auto& data_view) noexcept
#endif
    {
      using Scalar = collections::common_collection_type_t<decltype(data_view)>;
      using R = values::real_type_of_t<Scalar>;
      Scalar cf {2 * numbers::pi_v<R> / (max - min)};
      Scalar mid { R{max + min} * R{0.5}};
      Scalar a = collections::get(data_view, std::integral_constant<std::size_t, 0>{});
      return collections::views::generate(to_stat_collection {cf * (a - mid)}, std::integral_constant<std::size_t, 2>{});
    };


    /*
     * \details Maps x and y coordinates on Euclidean space back to an angle.
     */
    static constexpr auto
    from_stat_space =
#ifdef __cpp_concepts
    [](const T&, const collections::collection_view auto& data_view) noexcept
#else
    [](const T&, const auto& data_view) noexcept
#endif
    {
      using Scalar = collections::common_collection_type_t<decltype(data_view)>;
      using R = values::real_type_of_t<Scalar>;
      Scalar cf {2 * numbers::pi_v<R> / (max - min)};
      Scalar mid { R{max + min} * R{0.5}};
      Scalar x = collections::get(data_view, std::integral_constant<std::size_t, 0>{});
      Scalar y = collections::get(data_view, std::integral_constant<std::size_t, 1>{});
#ifdef __cpp_lib_ranges
      return std::views::single(values::atan2(y, x) / cf + mid);
#else
      return ranges::views::single(values::atan2(y, x) / cf + mid);
#endif
    };


  private:

#ifdef __cpp_concepts
    static constexpr auto
    wrap_impl(auto&& a) -> std::decay_t<decltype(a)>
#else
    template<typename Scalar>
    static constexpr std::decay_t<Scalar>
    wrap_impl(Scalar&& a)
#endif
    {
      auto ap = values::real(a);
      if (ap >= min and ap < max)
      {
        return std::forward<decltype(a)>(a);
      }
      else
      {
        using R = std::decay_t<decltype(ap)>;
        constexpr R period {max - min};
        using std::fmod;
        R ar {fmod(ap - R{min}, period)};
        if (ar < 0) return values::internal::update_real_part(std::forward<decltype(a)>(a), R{min} + ar + period);
        else return values::internal::update_real_part(std::forward<decltype(a)>(a), R{min} + ar);
      }
    }

  public:

    /*
     * \brief Return a collection_view that can get and update an angle wrapped into the primary range.
     */
    static constexpr auto
    wrap =
#ifdef __cpp_concepts
    [](const T&, collections::collection_view auto&& data_view) noexcept
#else
    [](const T&, auto&& data_view) noexcept
#endif
    {
      using D = decltype(data_view);
      using Scalar = collections::common_collection_type_t<decltype(data_view)>;
#ifdef __cpp_lib_ranges
      if constexpr (std::ranges::output_range<D, Scalar>)
#else
      if constexpr (ranges::output_range<D, Scalar>)
#endif
      {
        Scalar& a = collections::get(data_view, std::integral_constant<std::size_t, 0>{});
      }



      using V = decltype(data_view);
      auto wrap_get = [](V& v, auto i) { return wrap_impl(collections::get(v, std::move(i))); };
      auto wrap_set = [](V& v, auto i, auto x) -> auto&
      {
        auto& ret = collections::get(v, std::move(i));
        ret = wrap_impl(std::move(x));
        return ret;
      };
      return std::forward<decltype(data_view)>(data_view) | collections::views::update(wrap_get, wrap_set);
    };

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_ANGLE_HPP
