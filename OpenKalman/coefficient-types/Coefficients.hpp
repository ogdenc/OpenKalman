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
 * \brief Definitions for Coefficient class specializations and associated aliases.
 */

#ifndef OPENKALMAN_COEFFICIENTS_HPP
#define OPENKALMAN_COEFFICIENTS_HPP

#include <array>
#include <functional>
#include <numeric>

namespace OpenKalman
{
  /**
   * \brief An empty set of Coefficients.
   * \details This is a specialization of Coefficients.
   * \sa Coefficients
   */
  template<>
  struct Coefficients<>
  {
    /// Number of matrix rows corresponding to these coefficients.
    static constexpr std::size_t dimension = 0;


    /// Number of matrix rows when these coefficients are converted to Euclidean space.
    static constexpr std::size_t euclidean_dimension = 0;


    /// Whether all the coefficients are of type Axis.
    static constexpr bool axes_only = true;


    /// The type of the result when subtracting two Coefficients vectors.
    using difference_type = Coefficients<>;


    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    template<typename Scalar>
    using SetCoeff = std::function<void(const std::size_t, const Scalar)>;


    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);


    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, euclidean_dimension>
      to_euclidean_array = {};


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      from_euclidean_array = {};


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      wrap_array_get = {};


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, dimension>
      wrap_array_set = {};


    /**
     * \brief Prepend a set of new coefficients to the existing set.
     * \tparam Cnew The set of new coordinates to prepend.
     */
    template<typename ... Cnew>
    using Prepend = Coefficients<Cnew...>;


    /**
     * \brief Append a set of new coordinates to the existing set.
     * \tparam Cnew The set of new coordinates to append.
     */
    template<typename ... Cnew>
    using Append = Coefficients<Cnew...>;


    /**
     * \brief Extract a particular coefficient from the set of coefficients.
     * \tparam i The index of the coefficient.
     */
    template<std::size_t i>
    using Coefficient = Coefficients;


    /**
     * \brief Take the first <code>count</code> coefficients.
     * \tparam count The number of coefficients to take.
     */
    template<std::size_t count>
    using Take = Coefficients;


    /**
     * \brief Discard all remaining coefficients after the first <code>count</code>.
     * \tparam count The index of the first coefficient to discard.
     */
    template<std::size_t count>
    using Discard = Coefficients;


    static_assert(internal::coefficient_class<Coefficients>);
  };


  /**
   * \anchor CoefficientsCCs
   * \brief A set of two or more coefficient types.
   * \details This is a specialization of Coefficients.
   * \tparam C, Cs... The first and subsequent coefficient types.
   * \sa Coefficients
   */
#ifdef __cpp_concepts
  template<coefficients C, coefficients ... Cs>
#else
  template<typename C, typename ... Cs>
#endif
  struct Coefficients<C, Cs ...>
  {
#ifndef __cpp_concepts
    static_assert((coefficients<C> and ... and coefficients<Cs>));
#endif
    /// Number of matrix rows corresponding to these coefficients.
    static constexpr std::size_t dimension = C::dimension + Coefficients<Cs...>::dimension;


    /// Number of matrix rows when these coefficients are converted to Euclidean space.
    static constexpr std::size_t
    euclidean_dimension = C::euclidean_dimension + Coefficients<Cs...>::euclidean_dimension;


    /// Whether all the coefficients are of type Axis.
    static constexpr bool axes_only = C::axes_only and Coefficients<Cs...>::axes_only;


    /**
     * \brief The type of the result when subtracting two Coefficients vectors.
     * \details Each coefficient is subtracted independently.
     */
    using difference_type = Concatenate<typename C::difference_type, typename Cs::difference_type...>;


    /**
     * \brief A function taking a row index and returning a corresponding matrix element.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    /**
     * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    template<typename Scalar>
    using SetCoeff = std::function<void(const std::size_t, const Scalar)>;


    /**
     * \brief A pointer to a function (stored in an array) that takes a GetCoeff and returns a scalar value.
     */
    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);


    /**
     * \brief A pointer to a function (stored in an array) that takes a GetCoeff and returns a scalar value.
     */
    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);


    /**
     * \internal
     * \brief An array of functions that convert the coefficients to coordinates in Euclidean space.
     * \details The functions in the array take the coefficients and convert them to
     * Cartesian coordinates in a Euclidean space, depending on the type of each coordinate.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should generally be accessed only through \ref to_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first spherical coefficient that is being transformed.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, euclidean_dimension>
      to_euclidean_array = internal::join(C::template to_euclidean_array<Scalar, i>,
        Coefficients<Cs...>::template to_euclidean_array<Scalar, i + C::dimension>);


    /**
     * \internal
     * \brief An array of functions that convert coordinates in Euclidean space into the typed coordinates.
     * \details The functions in the array take Cartesian coordinates, and convert them to the typed coordinates.
     * The array element is a function taking a ''get coefficient'' function and returning the typed coordinates.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns one of
     * the Cartesian coordinates.
     * \note This should generally be accessed only through \ref internal::from_euclidean_coeff.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first of the Cartesian coordinates being transformed back to their respective types.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      from_euclidean_array = internal::join(C::template from_euclidean_array<Scalar, i>,
        Coefficients<Cs...>::template from_euclidean_array<Scalar, i + C::euclidean_dimension>);


    /**
     * \internal
     * \brief An array of functions that return a wrapped version of the coefficients.
     * \details Each function in the array takes a ''get coefficient'' function and returns wrapped coefficients.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns a coefficient.
     * \note This should generally be accessed only through \ref internal::wrap_get.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first of the coefficients that are being wrapped.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      wrap_array_get = internal::join(C::template wrap_array_get<Scalar, i>,
        Coefficients<Cs...>::template wrap_array_get<Scalar, i + C::dimension>);


    /**
     * \internal
     * \brief An array of functions that wraps and sets an existing matrix coefficient.
     * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
     * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
     * sets the coefficient at that index to a wrapped version of the scalar input.
     * \note This should generally be accessed only through \ref internal::wrap_set.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first of the coefficients that are being wrapped.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, dimension>
      wrap_array_set = internal::join(C::template wrap_array_set<Scalar, i>,
        Coefficients<Cs...>::template wrap_array_set<Scalar, i + C::dimension>);


    /**
     * \brief Prepend a set of new coefficients to the existing set.
     * \tparam Cnew The set of new coordinates to prepend.
     */
    template<typename ... Cnew>
    using Prepend = Coefficients<Cnew..., C, Cs ...>;


    /**
     * \brief Append a set of new coordinates to the existing set.
     * \tparam Cnew The set of new coordinates to append.
     */
    template<typename ... Cnew>
    using Append = Coefficients<C, Cs ..., Cnew ...>;


    /**
     * \brief Extract a particular coefficient from the set of coefficients.
     * \tparam i The index of the coefficient.
     */
    template<std::size_t i>
    using Coefficient = std::conditional_t<i == 0, C, typename Coefficients<Cs...>::template Coefficient<i - 1>>;


    /**
     * \brief Take the first <code>count</code> coefficients.
     * \tparam count The number of coefficients to take.
     */
    template<std::size_t count>
    using Take = std::conditional_t<count == 0,
      Coefficients<>,
      typename Coefficients<Cs...>::template Take<count - 1>::template Prepend<C>>;


    /**
     * \brief Discard all remaining coefficients after the first <code>count</code>.
     * \tparam count The index of the first coefficient to discard.
     */
    template<std::size_t count>
    using Discard = std::conditional_t<count == 0,
      Coefficients,
      typename Coefficients<Cs...>::template Discard<count - 1>>;


    static_assert(internal::coefficient_class<Coefficients>);
  };


  namespace detail
  {
    template<typename C, std::size_t N>
    struct ReplicateImpl
    {
      using type = typename ReplicateImpl<C, N - 1>::type::template Append<C>;
    };


    template<typename C>
    struct ReplicateImpl<C, 0>
    {
      using type = Coefficients<>;
    };


    template<typename...Cs, std::size_t N>
    struct ReplicateImpl<Coefficients<Cs...>, N>
    {
      using type = typename ReplicateImpl<Coefficients<Cs...>, N - 1>::type::template Append<Cs...>;
    };


    template<typename...Cs>
    struct ReplicateImpl<Coefficients<Cs...>, 0>
    {
      using type = Coefficients<>;
    };
  }


  /**
   * \brief Alias for <code>Coefficients<C...></code>, where <code>C</code> is repeated <var>N</var> times.
   * \tparam C The coefficient to be repeated.
   * \tparam N The number of times to repeat coefficient C.
   */
#ifdef __cpp_concepts
  template<coefficients C, std::size_t N>
#else
  template<typename C, std::size_t N>
#endif
  using Replicate = typename detail::ReplicateImpl<C, N>::type;


  /**
   * \brief Alias for <code>Coefficients<Axis...></code>, where Axis is repeated <code>dimension</code> times.
   * \tparam dimension The number of Axes.
   */
  template<std::size_t dimension>
  using Axes = Replicate<Axis, dimension>;


}// namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_HPP
