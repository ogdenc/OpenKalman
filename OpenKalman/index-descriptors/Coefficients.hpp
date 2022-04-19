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
   * \brief A set of coefficient types.
   * \details This is the key to the wrapping functionality of OpenKalman. Each of the fixed_coefficients Cs... matches-up with
   * one or more of the rows or columns of a matrix. The number of coefficients per coefficient depends on the dimension
   * of the coefficient. For example, Axis, Distance, Angle, and Inclination are dimension 1, and each correspond to a
   * single coefficient. Polar is dimension 2 and corresponds to two coefficients (e.g., a distance and an angle).
   * Spherical is dimension 3 and corresponds to three coefficients.
   * Example: <code>Coefficients&lt;Axis, angle::Radians&gt;</code>
   * \sa Specializations: Coefficients<>, \ref CoefficientsCCs "Coefficients<C, Cs...>"
   * \tparam Cs Any types within the concept coefficients.
   */
#ifdef __cpp_concepts
  template<fixed_coefficients...Cs>
#else
  template<typename...Cs>
#endif
  struct Coefficients;


  /**
   * \brief An empty set of Coefficients.
   * \details This is a specialization of Coefficients.
   * \sa Coefficients
   */
  template<>
  struct Coefficients<> : Dimensions<0>
  {
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    template<typename Scalar>
    using SetCoeff = std::function<void(const std::size_t, const Scalar)>;


    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);


    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, 0>
      to_euclidean_array = {};


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, 0>
      from_euclidean_array = {};


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, 0>
      wrap_array_get = {};


    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, 0>
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

  };


  /**
   * \anchor CoefficientsCCs
   * \brief A set of two or more coefficient types.
   * \details This is a specialization of Coefficients.
   * \tparam C, Cs... The first and subsequent coefficient types.
   * \sa Coefficients
   */
#ifdef __cpp_concepts
  template<fixed_coefficients C, fixed_coefficients ... Cs>
#else
  template<typename C, typename ... Cs>
#endif
  struct Coefficients<C, Cs ...> : Dimensions<dimension_size_of_v<C> + dimension_size_of_v<Coefficients<Cs...>>>
  {
#ifndef __cpp_concepts
    static_assert((fixed_coefficients<C> and ... and fixed_coefficients<Cs>));
#endif

  private:
    /// Number of matrix rows corresponding to these coefficients.
    static constexpr std::size_t dimension = (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>);


    /// Number of matrix rows when these coefficients are converted to Euclidean space.
    static constexpr std::size_t
    euclidean_dimension = (euclidean_dimension_size_of_v<C> + ... + euclidean_dimension_size_of_v<Cs>);

  public:

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
        Coefficients<Cs...>::template to_euclidean_array<Scalar, i + dimension_size_of_v<C>>);


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
        Coefficients<Cs...>::template from_euclidean_array<Scalar, i + euclidean_dimension_size_of_v<C>>);


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
        Coefficients<Cs...>::template wrap_array_get<Scalar, i + dimension_size_of_v<C>>);


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
        Coefficients<Cs...>::template wrap_array_set<Scalar, i + dimension_size_of_v<C>>);


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

  };


  /**
   * \internal
   * \brief Number of Euclidean dimensions is the sum of Euclidean dimensions of Cs.
   * \tparam Cs Component index descriptors
   */
#ifdef __cpp_concepts
  template<typename...Cs>
  struct euclidean_dimension_size_of<Coefficients<Cs...>>
#else
  template<typename...Cs>
  struct euclidean_dimension_size_of<Coefficients<Cs...>, std::enable_if_t<typed_index_descriptor<Coefficients<Cs...>>>>
#endif
    : std::integral_constant<std::size_t, (euclidean_dimension_size_of_v<Cs> + ... + 0)> {};


  /**
   * \internal
   * \brief The concatenation of the difference types of Cs.
   * \tparam Cs Component index descriptors
   */
  template<typename...Cs>
  struct dimension_difference_of<Coefficients<Cs...>>
  {
    using type = Concatenate<dimension_difference_of_t<Cs>...>;
  };


}// namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_HPP
