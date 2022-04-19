/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of Polar class and associated details.
 */

#ifndef OPENKALMAN_POLAR_HPP
#define OPENKALMAN_POLAR_HPP


namespace OpenKalman
{
  /**
   * \brief An atomic coefficient group reflecting polar coordinates.
   * \details C1 and C2 are coefficients, and must be some combination of Distance and Angle, such as
   * <code>Polar&lt;Distance, angle::Radians&gt; or Polar&lt;angle::Degrees, Distance&gt;</code>.
   * Polar coordinates span two adjacent coefficients in a matrix.
   * \tparam C1, C2 Distance and Angle, in either order. By default, they are Distance and angle::Radians, respectively.
   * \internal
   * <b>See also</b> the following functions for accessing coefficient properties:
   * - internal::to_euclidean_coeff: \copybrief internal::to_euclidean_coeff
   * - internal::from_euclidean_coeff: \copybrief internal::from_euclidean_coeff
   * - internal::wrap_get: \copybrief internal::wrap_get
   * - internal::wrap_set \copybrief internal::wrap_set
   */
#ifdef __cpp_concepts
  template<fixed_coefficients C1 = Distance, fixed_coefficients C2 = angle::Radians>
#else
  template<typename C1 = Distance, typename C2 = angle::Radians, typename Enable>
#endif
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

        if constexpr (std::numeric_limits<Scalar>::is_iec559)
          return std::atan2(y, x) / cf;
        else
        {
          if (x == 0)
            return 0;
          else
            return std::atan2(y, x) / cf;
        }
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get = {[](const GetCoeff& get_coeff)
      {
        return polar_angle_wrap_impl<Limits, Scalar>(std::signbit(get_coeff(i + d_i)), get_coeff(i + a_i));
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff&) {
            set_coeff(i + a_i, polar_angle_wrap_impl<Limits, Scalar>(false, s));
            // Assumes that the corresponding distance is positive.
          }
        };

    };


    // Implementation of polar coordinates.
    template<template<typename Scalar> typename Limits, typename Derived, typename C1, typename C2,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase : Dimensions<2>
    {
      /**
       * \internal
       * \brief A function taking a row index and returning a corresponding matrix element.
       * \details A separate function will be constructed for each column in the matrix.
       * \tparam Scalar The scalar type of the matrix.
       */
      template<typename Scalar>
      using GetCoeff = std::function<Scalar(const std::size_t)>;


      /**
       * \internal
       * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
       * \details A separate function will be constructed for each column in the matrix.
       * \tparam Scalar The scalar type of the matrix.
       */
      template<typename Scalar>
      using SetCoeff = std::function<void(const std::size_t, const Scalar)>;


      /**
       * \internal
       * \brief An array of functions that convert polar coordinates to coordinates in Euclidean space.
       * \details The functions in the array take the polar coordinates and convert them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder.
       * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
       * \note This should generally be accessed only through \ref to_euclidean_coeff.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first polar coefficient that is being transformed.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), 3>
        to_euclidean_array = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template to_euclidean_array<i, d_i, a_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template to_euclidean_array<i, d_i, a_i>
      );


      /**
       * \internal
       * \brief An array of functions that convert three coordinates in Euclidean space into a distance and angle.
       * \details The functions in the array take x, y, and z Cartesian coordinates representing a location on a
       * unit half-cylinder, and convert those coordinates to polar coordinates.
       * The array element is a function taking a ''get coefficient'' function and returning a distance or angle.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns one of the
       * three coordinates.
       * \note This should generally be accessed only through \ref internal::from_euclidean_coeff.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of the three Cartesian coordinates being transformed back to polar.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), 2>
        from_euclidean_array = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template from_euclidean_array<i, d2_i, x_i, y_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template from_euclidean_array<i, d2_i, x_i, y_i>
      );


      /**
       * \internal
       * \brief An array of functions that return a wrapped version of polar coordinates.
       * \details Each function in the array takes a ''get coefficient'' function and returns a distance and
       * wrapped angle.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns a coefficient.
       * \note This should generally be accessed only through \ref internal::wrap_get.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of two polar coordinates that are being wrapped.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), 2>
        wrap_array_get = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template wrap_array_get<i, d_i, a_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template wrap_array_get<i, d_i, a_i>
      );


      /**
       * \internal
       * \brief An array of functions that set a matrix coefficient to wrapped polar coordinates.
       * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
       * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
       * sets the coefficient at that index to a wrapped version of the scalar input.
       * \note This should generally be accessed only through \ref internal::wrap_set.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of the two polar coordinates that are being wrapped.
       */
      template<typename Scalar, std::size_t i>
      static constexpr
      std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), 2>
        wrap_array_set = internal::join(
        detail::PolarImpl<Limits, C1, Scalar>::template wrap_array_set<i, d_i, a_i>,
        detail::PolarImpl<Limits, C2, Scalar>::template wrap_array_set<i, d_i, a_i>
      );

    };

  } // namespace detail


  // (Radius, Angle).
  template<template<typename Scalar> typename Limits>
  struct Polar<Distance, Angle<Limits>>
    : detail::PolarBase<Limits, Polar<Distance, Angle<Limits>>, Distance, Angle<Limits>, 0, 1,  0, 1, 2> {};


  // (Angle, Radius).
  template<template<typename Scalar> typename Limits>
  struct Polar<Angle<Limits>, Distance>
    : detail::PolarBase<Limits, Polar<Angle<Limits>, Distance>, Angle<Limits>, Distance, 1, 0,  2, 0, 1> {};


  // Alternate cases:

#ifdef __cpp_concepts
  template<fixed_coefficients T1, fixed_coefficients T2>
  struct Polar<Coefficients<T1>, T2>
#else
  template<typename T1, typename T2>
  struct Polar<Coefficients<T1>, T2, std::enable_if_t<fixed_coefficients<T1> and fixed_coefficients<T2>>>
#endif
    : Polar<T1, T2> {};


#ifdef __cpp_concepts
  template<atomic_coefficient_group T1, fixed_coefficients T2>
  struct Polar<T1, Coefficients<T2>>
#else
  template<typename T1, typename T2>
  struct Polar<T1, Coefficients<T2>, std::enable_if_t<atomic_coefficient_group<T1> and fixed_coefficients<T2>>>
#endif
    : Polar<T1, T2> {};


  /**
    * \internal
    * \brief Polar is represented by three coordinates in Euclidean space.
    */
   template<typename T1, typename T2>
   struct euclidean_dimension_size_of<Polar<T1, T2>>
     : std::integral_constant<std::size_t, 3> {};


  /**
   * \internal
   * \brief The type of the result when subtracting two Polar vectors.
   * \details For differences, each coordinate behaves as if it were Distance or Angle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   */
  template<typename T1, typename T2>
  struct dimension_difference_of<Polar<T1, T2>>
  {
    using type = Concatenate<dimension_difference_of_t<T1>, dimension_difference_of_t<T2>>;
  };


}// namespace OpenKalman

#endif //OPENKALMAN_POLAR_HPP
