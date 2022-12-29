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
 * \brief Definition of the Angle class and related limits.
 */

#ifndef OPENKALMAN_ANGLE_HPP
#define OPENKALMAN_ANGLE_HPP

#include <array>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman
{
  template<typename Limits>
#ifdef __cpp_concepts
  requires floating_scalar_type<decltype(Limits::min)> and floating_scalar_type<decltype(Limits::max)> and
    (Limits::min < Limits::max) and (Limits::min <= 0) and (Limits::max > 0)
#endif
  struct Angle;


  /// Namespace for definitions relating to fixed_index_descriptors representing an angle.
  namespace angle
  {
    /**
     * \brief The numerical range [minimum, maximum)spanned by an angle.
     * \details The range include 0.
     * \tparam minimum The minimum angle (inclusive)
     * \tparam maximum The maximum angle (exclusive)
     */
    template<auto minimum, auto maximum>
    struct Limits
    {
      static constexpr auto min = minimum;
      static constexpr auto max = maximum;
    };


#if __cpp_nontype_template_args >= 201911L
    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<Limits<-numbers::pi_v<long double>, numbers::pi_v<long double>>>;
#else
    /// The limits of an angle measured in radians [-&pi;,&pi;).
    struct RadiansLimits
    {
      static constexpr long double min = -numbers::pi_v<long double>;
      static constexpr long double max = numbers::pi_v<long double>;
    };

    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<RadiansLimits>;
#endif


#if __cpp_nontype_template_args >= 201911L
    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<Limits<0, 2 * numbers::pi_v<long double>>>;
#else
    /// The limits of an angle measured in positive radians [0,2&pi;).
    struct PositiveRadiansLimits
    {
      static constexpr long double min = 0;
      static constexpr long double max = 2 * numbers::pi_v<long double>;
    };

    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<PositiveRadiansLimits>;
#endif


    /// An angle measured in degrees [0,360).
    using PositiveDegrees = Angle<Limits<0, 360>>;


    /// An angle measured in positive or negative degrees [-180,180).
    using Degrees = Angle<Limits<-180, 180>>;


    /// An wrapping circle such as the wrapping interval [0,1).
    using Circle = Angle<Limits<0, 1>>;

  } // namespace angle


  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [max,min) when it increases or decreases outside that range.
   * There are several predefined angles, including angle::Radians, angle::Degrees, angle::PositiveRadians,
   * angle::PositiveDegrees, and angle::Circle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam Limits A template class defining the real values <code>min</code> and <code>max</code>, representing
   * minimum and maximum values, respectively, beyond which wrapping occurs. This range must include 0.
   * <code>Scalar</code> is a \ref floating_scalar_type.
   */
#if __cpp_nontype_template_args >= 201911L
  template<typename Limits = angle::Limits<-numbers::pi_v<long double>, numbers::pi_v<long double>>>
#else
  template<typename Limits = angle::RadiansLimits>
#endif
#ifdef __cpp_concepts
    requires floating_scalar_type<decltype(Limits::min)> and floating_scalar_type<decltype(Limits::max)> and
      (Limits::min < Limits::max) and (Limits::min <= 0) and (Limits::max > 0)
#endif
  struct Angle
  {
#ifndef __cpp_concepts
    static_assert(floating_scalar_type<decltype(Limits::min)>);
    static_assert(floating_scalar_type<decltype(Limits::max)>);
    static_assert(Limits::min < Limits::max);
    static_assert(Limits::min <= 0);
    static_assert(Limits::max > 0);
#endif
  };


  namespace internal
  {
    template<typename T>
    struct is_angle_descriptor : std::false_type {};

    template<typename Limits>
    struct is_angle_descriptor<Angle<Limits>> : std::true_type {};
  }


  /**
   * \brief T is an index descriptor of an angle.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept angle_descriptor =
#else
  static constexpr bool angle_descriptor =
#endif
    internal::is_angle_descriptor<T>::value;


  namespace interface
  {
    /**
     * \internal
     * \brief traits for Angle.
     */
    template<typename Limits>
    struct FixedIndexDescriptorTraits<Angle<Limits>>
    {
      static constexpr std::size_t size = 1;
      static constexpr std::size_t euclidean_size = 2;
      static constexpr std::size_t component_count = 1;
      using difference_type = Angle<Limits>;
      static constexpr bool always_euclidean = false;

      /*
       * \details The angle corresponds to x and y coordinates on a unit circle.
       * By convention, the minimum angle limit Limits<Scalar::min corresponds to the point (-1,0) in Euclidean space,
       * and the angle is scaled so that the difference between Limits<Scalar>::min and Limits<<Scalar>::max is 2&pi;,
       * so Limits<Scalar>::max wraps back to the point (-1, 0).
       * \param euclidean_local_index A local index accessing either the x (if 0) or y (if 1) coordinate in Euclidean space
       */
  #ifdef __cpp_concepts
      static constexpr floating_scalar_type auto
      to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
      requires requires (std::size_t i){ {g(i)} -> floating_scalar_type; }
  #else
      template<typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
  #endif
      {
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
        const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
        const Scalar mid { R{Limits::max + Limits::min} * R{0.5}};

        Scalar theta = cf * (g(start) - mid); // Convert to radians
        if (euclidean_local_index == 0)
          return cosine(theta);
        else
          return sine(theta);
      }


      /*
       * \details The angle corresponds to x and y coordinates on a unit circle.
       * \param local_index This is assumed to be 0.
       * \param euclidean_start The starting location of the x and y coordinates within any larger set of index type descriptors
       */
  #ifdef __cpp_concepts
      static constexpr floating_scalar_type auto
      from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
      requires requires (std::size_t i){ {g(i)} -> floating_scalar_type; }
  #else
      template<typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
  #endif
      {
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
        const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
        const Scalar mid { R{Limits::max + Limits::min} * R{0.5}};

        Scalar x = g(euclidean_start);
        Scalar y = g(euclidean_start + 1);
        return arctangent2(y, x) / cf + mid;
      }


    private:

  #ifdef __cpp_concepts
      static constexpr auto wrap_impl(auto&& a) -> std::decay_t<decltype(a)>
  #else
      template<typename Scalar>
      static constexpr std::decay_t<Scalar> wrap_impl(Scalar&& a)
  #endif
      {
        auto ap = real_projection(a);
        if (not (ap < Limits::min) and ap < Limits::max)
        {
          return std::forward<decltype(a)>(a);
        }
        else
        {
          using R = std::decay_t<decltype(ap)>;
          constexpr R period {Limits::max - Limits::min};
          R ar {std::fmod(ap - R{Limits::min}, period)};
          if (ar < 0) return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::min} + ar + period);
          else return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::min} + ar);
        }
      }

    public:

      /*
       * \param local_index This is assumed to be 0.
       */
  #ifdef __cpp_concepts
      static constexpr floating_scalar_type auto
      wrap_get_element(const auto& g, std::size_t local_index, std::size_t start)
      requires requires (std::size_t i){ {g(i)} -> floating_scalar_type; }
  #else
      template<typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      wrap_get_element(const G& g, std::size_t local_index, std::size_t start)
  #endif
      {
        return wrap_impl(g(start));
      }


      /**
       * \param local_index This is assumed to be 0.
       */
  #ifdef __cpp_concepts
      static constexpr void
      wrap_set_element(const auto& s, const auto& g,
        const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
      requires requires (std::size_t i){ s(x, i); {x} -> floating_scalar_type; }
  #else
      template<typename S, typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type> and
        std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
      static constexpr void
      wrap_set_element(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
                       std::size_t local_index, std::size_t start)
  #endif
      {
        s(wrap_impl(x), start);
      }

    };


  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_ANGLE_HPP
