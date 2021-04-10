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
 * \brief Definition of the Inclination class and related limits.
 */

#ifndef OPENKALMAN_COEFFICIENTS_INCLINATION_H
#define OPENKALMAN_COEFFICIENTS_INCLINATION_H

#include <array>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman
{
  /// Namespace for definitions relating to coefficients representing an inclination.
  namespace inclination
  {
    /// Namespace for classes describing the numerical limits for an Inclination.
    namespace limits
    {
      /**
       * The limits of an inclination measured in radians: [-½&pi;,½&pi;].
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Radians
      {
        static constexpr Scalar up = std::numbers::pi_v<Scalar> / 2;
        static constexpr Scalar down = -std::numbers::pi_v<Scalar> / 2;
      };


      /**
       * The limits of an inclination measured in degrees: [-90,90].
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Degrees
      {
        static constexpr Scalar up = 90;
        static constexpr Scalar down = -90;
      };

    } // namespace limits

    /// An inclination measured in radians [-½&pi;,½&pi;].
    using Radians = Inclination<limits::Radians>;

    /// An inclination measured in degrees [-90,90].
    using Degrees = Inclination<limits::Degrees>;

  } // namespace inclination


  template<template<typename Scalar> typename Limits = inclination::limits::Radians>
#ifdef __cpp_concepts
    requires std::floating_point<decltype(Limits<double>::down)> and
      std::floating_point<decltype(Limits<double>::up)> and (Limits<double>::down < Limits<double>::up)
#endif
  struct Inclination
  {
    static_assert(Limits<double>::down < Limits<double>::up);

    /// Inclination is associated with one matrix element.
    static constexpr std::size_t size = 1;

    /// Inclination is represented by two coordinates in Euclidean space.
    static constexpr std::size_t euclidean_dimension = 2;

    /// Inclination is not composed of only axes.
    static constexpr bool axes_only = false;

    /**
     * \internal
     * A difference between two Inclination values does not wrap, and is treated as Axis.
     * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
     * 18th Int'l Conf. on Information Fusion 1550, 1555 (2015).
     */
    using difference_type = Coefficients<Axis>;

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
    static constexpr Scalar cf = std::numbers::pi_v<Scalar> / (Limits<Scalar>::up - Limits<Scalar>::down);

  public:
    /*
     * \internal
     * \brief An array of functions that convert an inclination coefficient to coordinates in Euclidean space.
     * \details The functions in the array each take the inclination angle and convert to one of the x or y coordinates
     * representing location in quadrants I or IV of the unit circle.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should be accessed only through \ref to_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the inclination coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), euclidean_dimension>
      to_euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return std::cos(get_coeff(i) * cf<Scalar>); },
        [](const GetCoeff<Scalar>& get_coeff) { return std::sin(get_coeff(i) * cf<Scalar>); }
      };


    /*
     * \internal
     * \brief An array of functions (here, just one) that convert coordinates in Euclidean space into an inclination.
     * \details The function in the array takes x and y coordinates representing a location in quadrants I or IV of the
     * unit circle, and convert those coordinates to an inclination angle.
     * The array element is a function taking a ''get coefficient'' function and returning an inclination.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns either
     * x (index i) or y (index i+1).
     * \note This should be accessed only through \ref internal::from_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the inclination coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff)
          {
            constexpr Scalar up = Limits<Scalar>::up;
            constexpr Scalar down = Limits<Scalar>::down;
            const auto x = get_coeff(i);
            const auto y = get_coeff(i + 1);
            if constexpr (up != -down)
            {
              return std::atan2(y, std::abs(x)) / cf<Scalar>;
            }
            else
            {
              constexpr Scalar range = up - down;
              constexpr Scalar period = range * 2;
              auto a = std::atan2(y, x) / cf<Scalar>;
              if (a < down) return down - a;
              if (a > range) a = down + period - a;
              return a;
            }
          }
      };


  private:
    template<typename Scalar>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    requires std::floating_point<Scalar>
#endif
    static Scalar wrap_impl(const Scalar s)
    {
      constexpr Scalar up = Limits<Scalar>::up;
      constexpr Scalar down = Limits<Scalar>::down;
      if (s >= down and s <= up)
      {
        return s;
      }
      else
      {
        constexpr Scalar range = up - down;
        constexpr Scalar period = range * 2;
        Scalar a = std::fmod(s - down, period);
        if (a < 0) a += period;
        if (a > range) a = period - a;
        return a + down;
      }
    }


  public:
    /*
     * \internal
     * \brief An array of functions (here, just one) that return a wrapped version of an inclination.
     * \details Each function in the array takes a ''get coefficient'' function and returning a non-wrapped inclination.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should be accessed only through \ref internal::wrap_get.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the inclination coefficient that is being wrapped.
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
     * \brief An array of functions (here, just one) that sets a matrix coefficient to a wrapped inclination.
     * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
     * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
     * sets the coefficient at that index to a wrapped version of the scalar input.
     * \note This should be accessed only through \ref internal::wrap_set.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the inclination coefficient that is being wrapped.
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


    static_assert(internal::coefficient_class<Inclination>);
  };

} // namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_H
