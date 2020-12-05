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
 * \brief Definition of Spherical class and associated details.
 */

#ifndef OPENKALMAN_SPHERICAL_H
#define OPENKALMAN_SPHERICAL_H


namespace OpenKalman
{
  template<typename C1 = Distance, typename C2 = angle::Radians, typename C3 = inclination::Radians>
  struct Spherical;


  namespace detail
  {
    template<template<typename Scalar> typename InclinationLimits, typename Scalar>
    static constexpr std::tuple<Scalar, bool>
    inclination_wrap_impl(const Scalar a)
    {
      constexpr Scalar max = InclinationLimits<Scalar>::up;
      constexpr Scalar min = InclinationLimits<Scalar>::down;
      constexpr Scalar range = max - min;
      constexpr Scalar period = 2 * range;
      if (a >= min and a <= max) // A shortcut, for the easy case.
      {
        return { a, false };
      }
      else
      {
        Scalar ar = std::fmod(a - min, period);
        if (ar < 0) ar += period;
        if (ar > range)
        {
          // Do a mirror reflection about vertical axis.
          return { period + min - ar, true };
        }
        else
        {
          return { min + ar, false };
        }
      }
    }


    template<template<typename Scalar> typename CircleLimits, typename Scalar>
    static inline Scalar azimuth_wrap_impl(const bool reflect_azimuth, const Scalar s)
    {
      constexpr Scalar max = CircleLimits<Scalar>::max;
      constexpr Scalar min = CircleLimits<Scalar>::min;
      constexpr Scalar period = max - min;

      Scalar a = reflect_azimuth ? s + period * 0.5 : s;

      if (a >= min and a < max) // Check if angle doesn't need wrapping.
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


    template<typename Coefficient, template<typename Scalar> typename CircleLimits,
      template<typename Scalar> typename InclinationLimits, typename Scalar>
    struct SphericalImpl;


    template<template<typename Scalar> typename CircleLimits, template<typename Scalar> typename InclinationLimits,
      typename Scalar>
    struct SphericalImpl<Distance, CircleLimits, InclinationLimits, Scalar>
    {
      static_assert(InclinationLimits<double>::up > InclinationLimits<double>::down);
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i, std::size_t z_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_euclidean_array = {[](const GetCoeff& get_coeff) constexpr {return std::abs(get_coeff(i + d_i)); }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get = {[](const GetCoeff& get_coeff) constexpr { return std::abs(get_coeff(i + d_i)); }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) {
            set_coeff(i + d_i, std::abs(s));
            if (std::signbit(s)) // If new distance is negative
            {
              set_coeff(i + a_i, azimuth_wrap_impl<CircleLimits>(true, get_coeff(i + a_i))); // Adjust azimuth.
              set_coeff(i + i_i, -get_coeff(i + i_i)); // Adjust inclination.
            }
          }
        };

    };

    template<template<typename Scalar> typename CircleLimits, template<typename Scalar> typename InclinationLimits,
      typename Scalar>
    struct SphericalImpl<Angle<CircleLimits>, CircleLimits, InclinationLimits, Scalar>
    {
      static_assert(InclinationLimits<double>::up > InclinationLimits<double>::down);
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

      static constexpr Scalar cf_cir = 2 * std::numbers::pi_v<Scalar> /
        (CircleLimits<Scalar>::max - CircleLimits<Scalar>::min);

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i, std::size_t z_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
        const auto x = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + x_i) : get_coeff(i + x_i);
        const auto y = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + y_i) : get_coeff(i + y_i);
        return std::atan2(y, x) / cf_cir;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get = {[](const GetCoeff& get_coeff) {
          const auto [new_i, b] = inclination_wrap_impl<InclinationLimits>(get_coeff(i + i_i));
          const bool reflect_azimuth = b != std::signbit(get_coeff(i + d_i));
          return azimuth_wrap_impl<CircleLimits>(reflect_azimuth, get_coeff(i + a_i));
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff&) {
            set_coeff(i + a_i, azimuth_wrap_impl<CircleLimits>(false, s)); // Assume distance and inclination are correct.
          }
        };

    };

    template<template<typename Scalar> typename CircleLimits, template<typename Scalar> typename InclinationLimits,
      typename Scalar>
    struct SphericalImpl<Inclination<InclinationLimits>, CircleLimits, InclinationLimits, Scalar>
    {
      static_assert(InclinationLimits<double>::up > InclinationLimits<double>::down);
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

      static constexpr Scalar cf_inc = std::numbers::pi_v<Scalar> /
        (InclinationLimits<Scalar>::up - InclinationLimits<Scalar>::down);

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i, std::size_t z_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
        const auto r = std::hypot(get_coeff(i + x_i), get_coeff(i + y_i), get_coeff(i + z_i));
        const auto ret = std::asin(get_coeff(i + z_i) / r) / cf_inc;
        if (std::isnan(ret)) return 0.; // Avoid NAN when all coefficients are zero.
        else return std::signbit(get_coeff(i + d_i)) ? -ret : ret;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get =  {[](const GetCoeff& get_coeff) constexpr {
        const auto [new_i, b] = inclination_wrap_impl<InclinationLimits>(get_coeff(i + i_i));
        return std::signbit(get_coeff(i + d_i)) ? -new_i : new_i;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) {
            const auto [new_i, b] = inclination_wrap_impl<InclinationLimits>(s);
            set_coeff(i + i_i, new_i); // Adjust inclination.
            set_coeff(i + a_i, azimuth_wrap_impl<CircleLimits>(b, get_coeff(i + a_i))); // Adjust azimuth.
          }
        };

    };

    // Implementation of polar coordinates.
    template<typename Derived,
      typename C1, typename C2, typename C3,
      template<typename Scalar> typename CircleLimits, template<typename Scalar> typename InclinationLimits,
      std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
      /// Spherical is associated with three matrix elements.
      static constexpr std::size_t size = 3;

      /// Spherical is represented by four coordinates in Euclidean space.
      static constexpr std::size_t dimension = 4;

      /// Spherical is not composed of only axes.
      static constexpr bool axes_only = false;

      /**
       * \brief The type of the result when subtracting two Spherical vectors.
       * \details For differences, each coordinate behaves as if it were Distance, Angle, or Inclination.
       * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
       * 18th Int'l Conf. on Information Fusion 1550, 1555 (2015).
       */
      using difference_type =
        Concatenate<typename C1::difference_type, typename C2::difference_type, typename C3::difference_type>;


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
      static constexpr Scalar cf_cir = 2 * std::numbers::pi_v<Scalar> /
        (CircleLimits<Scalar>::max - CircleLimits<Scalar>::min);

      template<typename Scalar>
      static constexpr Scalar cf_inc = std::numbers::pi_v<Scalar> /
        (InclinationLimits<Scalar>::up - InclinationLimits<Scalar>::down);

    public:
      /*
       * \internal
       * \brief An array of functions that convert spherical coordinates to coordinates in Euclidean space.
       * \details The functions in the array take the spherical coordinates and convert them to four
       * Cartesian coordinates representing a distance and a location on unit sphere.
       * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
       * \note This should be accessed only through \ref internal::to_euclidean_coeff.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first spherical coefficient that is being transformed.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
        to_euclidean_array = {
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return get_coeff(i + d_i); },
        [](const GetCoeff<Scalar>& get_coeff) constexpr {
          return std::cos(get_coeff(i + a_i) * cf_cir<Scalar>) * std::cos(get_coeff(i + i_i) * cf_inc<Scalar>);
        },
        [](const GetCoeff<Scalar>& get_coeff) constexpr {
          return std::sin(get_coeff(i + a_i) * cf_cir<Scalar>) * std::cos(get_coeff(i + i_i) * cf_inc<Scalar>);
        },
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::sin(get_coeff(i + i_i) * cf_inc<Scalar>); }
      };


      /*
       * \internal
       * \brief An array of functions that convert four coordinates in Euclidean space into spherical coordinates.
       * \details The functions in the array take four Cartesian coordinates representing a distance and a location
       * on a unit sphere, and convert those coordinates to spherical coordinates.
       * The array element is a function taking a ''get coefficient'' function and returning spherical coordinates.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns one of
       * the four coordinates.
       * \note This should be accessed only through \ref internal::from_euclidean_coeff.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of the four Cartesian coordinates being transformed back to spherical.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        from_euclidean_array = internal::join(
        internal::join(
          detail::SphericalImpl<C1, CircleLimits, InclinationLimits, Scalar>::template from_euclidean_array<i, 0, 1, 2, 3>,
          detail::SphericalImpl<C2, CircleLimits, InclinationLimits, Scalar>::template from_euclidean_array<i, 0, 1, 2, 3>),
          detail::SphericalImpl<C3, CircleLimits, InclinationLimits, Scalar>::template from_euclidean_array<i, 0, 1, 2, 3>
      );


      /**
       * \internal
       * \brief An array of functions that return a wrapped version of spherical coordinates.
       * \details Each function in the array takes a ''get coefficient'' function and returns spherical coordinates.
       * The ''get coefficient'' function takes the index of a column within a row vector and returns a coefficient.
       * \note This should be accessed only through \ref internal::wrap_get.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of three spherical coordinates that are being wrapped.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        wrap_array_get = internal::join(
        internal::join(
          detail::SphericalImpl<C1, CircleLimits, InclinationLimits, Scalar>::template wrap_array_get<i, d_i, a_i, i_i>,
          detail::SphericalImpl<C2, CircleLimits, InclinationLimits, Scalar>::template wrap_array_get<i, d_i, a_i, i_i>),
          detail::SphericalImpl<C3, CircleLimits, InclinationLimits, Scalar>::template wrap_array_get<i, d_i, a_i, i_i>
      );


      /**
       * \internal
       * \brief An array of functions that set a matrix coefficient to wrapped spherical coordinates.
       * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
       * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
       * sets the coefficient at that index to a wrapped version of the scalar input.
       * \note This should be accessed only through \ref internal::wrap_set.
       * \tparam Scalar The scalar type (e.g., double).
       * \tparam i The index of the first of the three spherical coordinates that are being wrapped.
       */
      template<typename Scalar, std::size_t i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
        wrap_array_set = internal::join(
        internal::join(
          detail::SphericalImpl<C1, CircleLimits, InclinationLimits, Scalar>::template wrap_array_set<i, d_i, a_i, i_i>,
          detail::SphericalImpl<C2, CircleLimits, InclinationLimits, Scalar>::template wrap_array_set<i, d_i, a_i, i_i>),
          detail::SphericalImpl<C3, CircleLimits, InclinationLimits, Scalar>::template wrap_array_set<i, d_i, a_i, i_i>
      );

    };

  } // namespace detail


  // Distance, Angle, Inclination.
  template<template<typename Scalar> typename ALimits, template<typename Scalar> typename ILimits>
  struct Spherical<Distance, Angle<ALimits>, Inclination<ILimits>>
    : detail::SphericalBase<Spherical<Distance, Angle<ALimits>, Inclination<ILimits>>,
    Distance, Angle<ALimits>, Inclination<ILimits>, ALimits, ILimits, 0, 1, 2>
  {
    static_assert(internal::coefficient_class<Spherical>);
  };


  // Distance, Inclination, Angle.
  template<template<typename Scalar> typename ILimits, template<typename Scalar> typename ALimits>
  struct Spherical<Distance, Inclination<ILimits>, Angle<ALimits>>
    : detail::SphericalBase<Spherical<Distance, Inclination<ILimits>, Angle<ALimits>>,
    Distance, Inclination<ILimits>, Angle<ALimits>, ALimits, ILimits, 0, 2, 1>
  {
  static_assert(internal::coefficient_class<Spherical>);
  };


  // Angle, Distance, Inclination.
  template<template<typename Scalar> typename ALimits, template<typename Scalar> typename ILimits>
  struct Spherical<Angle<ALimits>, Distance, Inclination<ILimits>>
    : detail::SphericalBase<Spherical<Angle<ALimits>, Distance, Inclination<ILimits>>,
      Angle<ALimits>, Distance, Inclination<ILimits>, ALimits, ILimits, 1, 0, 2>
  {
    static_assert(internal::coefficient_class<Spherical>);
  };


  // Inclination, Distance, Angle.
  template<template<typename Scalar> typename ILimits, template<typename Scalar> typename ALimits>
  struct Spherical<Inclination<ILimits>, Distance, Angle<ALimits>>
    : detail::SphericalBase<Spherical<Inclination<ILimits>, Distance, Angle<ALimits>>,
      Inclination<ILimits>, Distance, Angle<ALimits>, ALimits, ILimits, 1, 2, 0>
  {
    static_assert(internal::coefficient_class<Spherical>);
  };


  // Angle, Inclination, Distance.
  template<template<typename Scalar> typename ALimits, template<typename Scalar> typename ILimits>
  struct Spherical<Angle<ALimits>, Inclination<ILimits>, Distance>
    : detail::SphericalBase<Spherical<Angle<ALimits>, Inclination<ILimits>, Distance>,
      Angle<ALimits>, Inclination<ILimits>, Distance, ALimits, ILimits, 2, 0, 1>
  {
    static_assert(internal::coefficient_class<Spherical>);
  };


  // Inclination, Angle, Distance.
  template<template<typename Scalar> typename ILimits, template<typename Scalar> typename ALimits>
  struct Spherical<Inclination<ILimits>, Angle<ALimits>, Distance>
    : detail::SphericalBase<Spherical<Inclination<ILimits>, Angle<ALimits>, Distance>,
      Inclination<ILimits>, Angle<ALimits>, Distance, ALimits, ILimits, 2, 1, 0>
  {
    static_assert(internal::coefficient_class<Spherical>);
  };


}// namespace OpenKalman

#endif //OPENKALMAN_SPHERICAL_H
