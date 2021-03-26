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

#ifndef OPENKALMAN_ANGLE_H
#define OPENKALMAN_ANGLE_H

#include <array>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman
{
  /// Namespace for definitions relating to coefficients representing an angle.
  namespace angle
  {
    /// Namespace for classes describing the numerical limits for an angle.
    namespace limits
    {
      /**
       * The limits of an angle measured in radians: [-&pi;,&pi;).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Radians
      {
        static constexpr Scalar max = std::numbers::pi_v<Scalar>;
        static constexpr Scalar min = -std::numbers::pi_v<Scalar>;
      };


      /**
       * The limits of an angle measured in positive radians: [0,2&pi;).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct PositiveRadians
      {
        static constexpr Scalar max = 2 * std::numbers::pi_v<Scalar>;
        static constexpr Scalar min = 0;
      };


      /**
       * The limits of an angle measured in positive or negative degrees: [-180,180).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Degrees
      {
        static constexpr Scalar max = 180;
        static constexpr Scalar min = -180;
      };


      /**
       * The limits of an angle measured in positive degrees: [0,360).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct PositiveDegrees
      {
        static constexpr Scalar max = 360;
        static constexpr Scalar min = 0;
      };


      /**
       * The limits of a wrapping circle, such as the wrapping interval [0,1).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       * \tparam a_min The minimum value before wrapping. (Available in c++20+).
       * \tparam a_max The maximum value before wrapping. (Available in c++20+).
       */
#if __cpp_nontype_template_args >= 201911L
      template<typename Scalar = double, Scalar a_min = 0, Scalar a_max = 1>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Circle
      {
        static constexpr Scalar min = a_min;
        static constexpr Scalar max = a_max;
      };
#else
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Circle
      {
        static constexpr Scalar min = 0;
        static constexpr Scalar max = 1;
      };
#endif

    } // namespace limits

    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<limits::Radians>;

    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<limits::PositiveRadians>;

    /// An angle measured in degrees [0,360).
    using PositiveDegrees = Angle<limits::PositiveDegrees>;

    /// An angle measured in positive or negative degrees [-180,180).
    using Degrees = Angle<limits::Degrees>;

    /// An wrapping circle such as the wrapping interval [0,1).
    using Circle = Angle<limits::Circle>;

  } // namespace angle


  template<template<typename Scalar> typename Limits = angle::limits::Radians>
#ifdef __cpp_concepts
    requires std::floating_point<decltype(Limits<double>::min)> and
      std::floating_point<decltype(Limits<double>::max)> and (Limits<double>::min < Limits<double>::max)
#endif
  struct Angle
  {
    static_assert(Limits<double>::min < Limits<double>::max);

    /// Angle is associated with one matrix element.
    static constexpr std::size_t size = 1;

    /// Angle is represented by two coordinates in Euclidean space.
    static constexpr std::size_t dimension = 2;

    /// Angle is not composed of only axes.
    static constexpr bool axes_only = false;

    /**
     * \brief The type of the result when subtracting two Angle values.
     * \details A distance between two points on a circle cannot be more than the circumference of the circle,
     * so it must be wrapped as an Angle.
     * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
     * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
     */
    using difference_type = Angle;

    /*
     * \internal
     * \brief A function taking a row index and returning a corresponding matrix element.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    /*
     * \internal
     * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    template<typename Scalar>
    using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

  private:
    template<typename Scalar>
    static constexpr Scalar cf = 2 * std::numbers::pi_v<Scalar> / (Limits<Scalar>::max - Limits<Scalar>::min);

  public:
    /*
     * \internal
     * \brief An array of functions that convert an angle coefficient to coordinates in Euclidean space.
     * \details The functions in the array each take the angle and convert it to one of the x or y coordinates
     * representing a location on a unit circle.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should be accessed only through \ref to_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the angle coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
      to_euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return std::cos(get_coeff(i) * cf<Scalar>); },
        [](const GetCoeff<Scalar>& get_coeff) { return std::sin(get_coeff(i) * cf<Scalar>); }
      };


    /*
     * \internal
     * \brief An array of functions (here, just one) that convert coordinates in Euclidean space into an angle.
     * \details The function in the array takes x and y coordinates representing a location on a
     * unit circle, and convert those coordinates to an angle.
     * The array element is a function taking a ''get coefficient'' function and returning an angle.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns either
     * x (index i) or y (index i+1).
     * \note This should be accessed only through \ref internal::from_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index within the Euclidean matrix of the x coefficient corresponding to the angle.
     * The index of the corresponding y coefficient is i+1.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) {
          constexpr Scalar max = Limits<Scalar>::max;
          constexpr Scalar min = Limits<Scalar>::min;
          auto a = std::atan2(get_coeff(i + 1), get_coeff(i)) / cf<Scalar>;
          if constexpr (max != -min)
          {
            constexpr Scalar period = max - min;
            if (a < min) return a + period; // Generally, this is for positive angle systems where min is 0.
          }
          return a;
        }
      };


  private:
    template<typename Scalar>
    static Scalar wrap_impl(const Scalar a)
    {
      constexpr Scalar max = Limits<Scalar>::max;
      constexpr Scalar min = Limits<Scalar>::min;
      if (a >= min and a < max)
      {
        return a;
      }
      else
      {
        constexpr Scalar period = max - min;
        Scalar ar = std::fmod(a - min, period);
        if (ar < 0) ar += period;
        return ar + min;
      }
    }

  public:
    /*
     * \internal
     * \brief An array of functions (here, just one) that return a wrapped version of an angle.
     * \details Each function in the array takes a ''get coefficient'' function and returns a wrapped angle.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should be accessed only through \ref internal::wrap_get.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the angle coefficient that is being wrapped.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array_get =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return wrap_impl(get_coeff(i)); }
      };


    /*
     * \internal
     * \brief An array of functions (here, just one) that set a matrix coefficient to a wrapped angle.
     * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
     * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
     * sets the coefficient at that index to a wrapped version of the scalar input.
     * \note This should be accessed only through \ref internal::wrap_set.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the angle coefficient that is being wrapped.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(i, wrap_impl(s)); }
      };


    static_assert(internal::coefficient_class<Angle>);
  };


} // namespace OpenKalman


#endif //OPENKALMAN_ANGLE_H
