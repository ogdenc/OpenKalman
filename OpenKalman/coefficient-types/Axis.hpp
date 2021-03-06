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
 * \brief Definition of the Axis class.
 */

#ifndef OPENKALMAN_AXIS_HPP
#define OPENKALMAN_AXIS_HPP

#include <cmath>
#include <array>
#include <functional>

namespace OpenKalman
{
  struct Axis
  {
    /// Axis is associated with one matrix element.
    static constexpr std::size_t dimensions = 1;

    /// Axis is represented by one coordinate in Euclidean space.
    static constexpr std::size_t euclidean_dimensions = 1;

    /// Axis is composed of only axes.
    static constexpr bool axes_only = true;

    /**
     * \brief The type of the result when subtracting two Axis values.
     * \details A difference between two Axis values is also on an Axis, so there is no wrapping.
     */
    using difference_type = Axis;


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
     * \brief An array of functions (here, just one) that transform an axis coefficient to Euclidean space.
     * \details Because an axis already represents a point in Euclidean space, this is an identity function.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should generally be accessed only through \ref to_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), euclidean_dimensions>
    to_euclidean_array =
    {
      [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
    };


    /**
     * \internal
     * \brief An array of functions (here, just one) that transform a coordinate in Euclidean space into an axis.
     * \details Because an axis already represents a point in Euclidean space, this is an identity function.
     * The array element is a function taking a ''get coefficient'' function and returning an axis.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the value.
     * \note This should generally be accessed only through \ref internal::from_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimensions>
      from_euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };


    /**
     * \internal
     * \brief An array of functions (here, just one) that return a wrapped version of an axis.
     * \details Each function in the array takes a ''get coefficient'' function and returning an (unchanged) axis.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should generally be accessed only through \ref internal::wrap_get.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the axis coefficient that is being wrapped.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimensions>
      wrap_array_get =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };


    /**
     * \internal
     * \brief An array of functions (here, just one) that sets a matrix coefficient to an axis value.
     * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
     * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
     * sets the coefficient at that index to that (unchanged) scalar input.
     * \note This should generally be accessed only through \ref internal::wrap_set.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the axis coefficient that is being wrapped.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr
      std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), dimensions>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(i, s); }
      };


    static_assert(internal::coefficient_class<Axis>);
  };


} // namespace OpenKalman


#endif //OPENKALMAN_AXIS_HPP
