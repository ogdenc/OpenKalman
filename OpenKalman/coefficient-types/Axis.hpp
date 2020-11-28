/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_AXIS_H
#define OPENKALMAN_AXIS_H

#include <cmath>
#include <array>
#include <functional>

namespace OpenKalman
{
  struct Axis
  {
    static constexpr std::size_t size = 1; ///< The coefficient represents 1 coordinate.
    static constexpr std::size_t dimension = 1; ///< The coefficient represents 1 coordinate in Euclidean space.
    static constexpr bool axes_only = true; ///< The coefficients is Axis.

    /**
     * \brief The type of the result when subtracting two Axis values.
     * \details A difference between two Axis values is also on an Axis, so there is no wrapping.
     */
    using difference_type = Axis;

  private:
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;


  public:
    /**
     * \internal
     * \brief An array of functions (here, just one) that transform an axis coefficient to Euclidean space.
     * \details Because an axis already represents a point in Euclidean space, this is an identity function.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
      to_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };


    /**
     * \internal
     * \brief An array of functions (here, just one) that transform a coordinate in Euclidean space into an axis.
     * \details Because an axis already represents a point in Euclidean space, this is an identity function.
     * The array element is a function taking a ''get coefficient'' function and returning an axis.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the value.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the coefficient that is being transformed.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };


    /**
     * \internal
     * \brief An array of functions (here, just one) that return a wrapped version of an axis.
     * \details Each function in the array takes a ''get coefficient'' function and returning an (unchanged) axis.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the axis coefficient that is being wrapped.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
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
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the axis coefficient that is being wrapped.
     */
    template<typename Scalar, std::size_t i>
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
      requires std::is_arithmetic_v<Scalar>
#endif
    static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(s, i); }
      };

    static_assert(internal::coefficient_class<Axis>);
  };

} // namespace OpenKalman


#endif //OPENKALMAN_AXIS_H
