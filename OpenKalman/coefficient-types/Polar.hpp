/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of Polar class and associated details.
 */

#ifndef OPENKALMAN_POLAR_H
#define OPENKALMAN_POLAR_H


namespace OpenKalman
{
  template<typename C1 = Distance, typename C2 = angle::Radians>
  struct Polar;


  namespace detail
  {
    template<template<typename Scalar> typename Limits, typename Scalar>
    static inline Scalar polar_angle_wrap_impl(const bool distance_is_negative, const Scalar s)
    {
      constexpr Scalar max = Limits<Scalar>::max;
      constexpr Scalar min = Limits<Scalar>::min;
      constexpr Scalar period = max - min;

      Scalar a = distance_is_negative ? s + period * 0.5 : s;

      if (a >= min and a < max) // Check if the angle doesn't need wrapping.
      {
        return a;
      }
      else // Wrap the angle.
      {
        Scalar ar = std::fmod(a - min, period);
        if (ar < 0)
        {
          ar += period;
        }
        return ar + min;
      }
    }


    template<template<typename Scalar> typename Limits, typename Coefficient, typename Scalar>
    struct PolarImpl;


    template<template<typename Scalar> typename Limits, typename Scalar>
    struct PolarImpl<Limits, Distance, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        to_euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
          return get_coeff(i + d_i);
        }};

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
          return std::abs(get_coeff(i + d_i));
        }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get =
        {
          [](const GetCoeff& get_coeff) { return std::abs(get_coeff(i + d_i)); }
        };

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) {
            set_coeff(i + d_i, std::abs(s));
            const auto a = get_coeff(i + a_i);
            set_coeff(i + a_i, polar_angle_wrap_impl<Limits>(std::signbit(s), a)); // May need to reflect angle.
          }
        };

    };


    template<template<typename Scalar> typename Limits, typename Scalar>
    struct PolarImpl<Limits, Angle<Limits>, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

      static constexpr Scalar cf = 2 * std::numbers::pi_v<Scalar> / (Limits<Scalar>::max - Limits<Scalar>::min);

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 2>
        to_euclidean_array =
        {
          [](const GetCoeff& get_coeff) { return std::cos(get_coeff(i + a_i) * cf); },
          [](const GetCoeff& get_coeff) { return std::sin(get_coeff(i + a_i) * cf); }
        };

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_euclidean_array = {[](const GetCoeff& get_coeff)
      {
        const auto x = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + x_i) : get_coeff(i + x_i);
        const auto y = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + y_i) : get_coeff(i + y_i);
        return std::atan2(y, x) / cf;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get = {[](const GetCoeff& get_coeff)
      {
        return polar_angle_wrap_impl<Limits>(std::signbit(get_coeff(i + d_i)), get_coeff(i + a_i));
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff&) {
            set_coeff(i + a_i, polar_angle_wrap_impl<Limits>(false, s));
            // Assumes that the corresponding distance is positive.
          }
        };

    };


    // Implementation of polar coordinates.
    template<template<typename Scalar> typename Limits, typename Derived, typename C1, typename C2,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
      /// Polar is associated with two matrix elements.
      static constexpr std::size_t size = 2;

      /// Polar is represented by three coordinates in Euclidean space.
      static constexpr std::size_t dimension = 3;

      /// Polar is not composed of only axes.
      static constexpr bool axes_only = false;

      /**
       * \brief The type of the result when subtracting two Polar vectors.
       * \details For differences, each coordinate behaves as if it were Distance or Angle.
       * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
       * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
       */
      using difference_type = Concatenate<typename C1::difference_type, typename C2::difference_type>;


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


      /*
       * \internal
       * \brief An array of functions that convert polar coordinates to coordinates in Euclidean space.
       * \details The functions in the array take the polar coordinates and convert them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder.
       * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
       * \note This should be accessed only through \ref to_euclidean_coeff.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first polar coefficient that is being transformed.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
        to_euclidean_array = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template to_euclidean_array<i, d_i, a_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template to_euclidean_array<i, d_i, a_i>
      );


      /*
       * \internal
       * \brief An array of functions that convert three coordinates in Euclidean space into a distance and angle.
       * \details The functions in the array take x, y, and z Cartesian coordinates representing a location on a
       * unit half-cylinder, and convert those coordinates to polar coordinates.
       * The array element is a function taking a ''get coefficient'' function and returning a distance or angle.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns one of the
       * three coordinates.
       * \note This should be accessed only through \ref internal::from_euclidean_coeff.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of the three Cartesian coordinates being transformed back to polar.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        from_euclidean_array = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template from_euclidean_array<i, d2_i, x_i, y_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template from_euclidean_array<i, d2_i, x_i, y_i>
      );


      /*
       * \internal
       * \brief An array of functions that return a wrapped version of polar coordinates.
       * \details Each function in the array takes a ''get coefficient'' function and returns a distance and
       * wrapped angle.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns a coefficient.
       * \note This should be accessed only through \ref internal::wrap_get.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of two polar coordinates that are being wrapped.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        wrap_array_get = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template wrap_array_get<i, d_i, a_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template wrap_array_get<i, d_i, a_i>
      );


      /*
       * \internal
       * \brief An array of functions that set a matrix coefficient to wrapped polar coordinates.
       * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
       * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
       * sets the coefficient at that index to a wrapped version of the scalar input.
       * \note This should be accessed only through \ref internal::wrap_set.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of the two polar coordinates that are being wrapped.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
        wrap_array_set = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template wrap_array_set<i, d_i, a_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template wrap_array_set<i, d_i, a_i>
      );

    };

  } // namespace detail


  // (Radius, Angle).
  template<template<typename Scalar> typename Limits>
  struct Polar<Distance, Angle<Limits>>
    : detail::PolarBase<Limits, Polar<Distance, Angle<Limits>>, Distance, Angle<Limits>, 0, 1,  0, 1, 2>
  {
    static_assert(internal::coefficient_class<Polar>);
  };


  // (Angle, Radius).
  template<template<typename Scalar> typename Limits>
  struct Polar<Angle<Limits>, Distance>
    : detail::PolarBase<Limits, Polar<Angle<Limits>, Distance>, Angle<Limits>, Distance, 1, 0,  2, 0, 1>
  {
    static_assert(internal::coefficient_class<Polar>);
  };

}// namespace OpenKalman

#endif //OPENKALMAN_POLAR_H
