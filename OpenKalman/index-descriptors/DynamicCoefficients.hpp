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
 * \brief Definition of the DynamicCoefficients class.
 */

#ifndef OPENKALMAN_DYNAMICCOEFFICIENTS_HPP
#define OPENKALMAN_DYNAMICCOEFFICIENTS_HPP

#include <vector>
#include <functional>
#include <typeindex>

namespace OpenKalman
{
  /**
   * \brief A list of coefficients defined at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \internal
   * <b>See also</b> the following functions for accessing coefficient properties:
   * - internal::to_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff):
   * \copybrief internal::to_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff)
   * - internal::from_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff):
   * \copybrief internal::from_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff)
   * - internal::wrap_get(Coeffs&& coeffs, const std::size_t row, const F& get_coeff):
   * \copybrief internal::wrap_get(Coeffs&& coeffs, const std::size_t row, const F& get_coeff)
   * - internal::wrap_set(Coeffs&& coeffs, const std::size_t row, const Scalar s, const FS& set_coeff,
   * const FG& get_coeff)
   * \copybrief internal::wrap_set(Coeffs&& coeffs, const std::size_t row, const Scalar s, const FS& set_coeff,
   * const FG& get_coeff)
   */
  template<typename Scalar = double>
  struct DynamicCoefficients;


  template<typename Scalar_>
  struct DynamicCoefficients : Dimensions<dynamic_size>
  {
    /// The scalar type of the coefficients
    using Scalar = Scalar_;

    /// The number of dimension at runtime.
    const std::size_t runtime_dimension;

    /// The number of coordinates in Euclidean space at runtime.
    const std::size_t runtime_euclidean_dimension;

    /**
     * \brief The type of the result when subtracting two DynamicCoefficients values.
     * \details A difference between two dynamic coefficients is also dynamic.
     */
    using difference_type = DynamicCoefficients;

    /// The type index of the corresponding \ref fixed_coefficients.
    const std::type_index id;


    /**
     * \internal
     * \brief A function taking a row index and returning a corresponding matrix element.
     * \details A separate function will be constructed for each column in the matrix.
     */
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    /**
     * \internal
     * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
     * \details A separate function will be constructed for each column in the matrix.
     */
    using SetCoeff = std::function<void(const std::size_t, const Scalar)>;

    /**
     * \internal
     * \brief Get a coordinate in Euclidean space corresponding to a coefficient in a matrix with typed coefficients.
     * \param row The applicable row of the transformed matrix.
     * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix
     * and returning its scalar value.
     * \return The scalar value of the transformed coordinate in Euclidean space corresponding to the provided row.
     */
    std::function<Scalar(const std::size_t row, const GetCoeff& get_coeff)> to_euclidean_coeff;


    /**
     * \internal
     * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
     * \param row The applicable row of the transformed matrix.
     * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
     * returning its scalar value.
     * \return The scalar value of the typed coefficient corresponding to the provided row.
     */
    std::function<Scalar(const std::size_t row, const GetCoeff& get_coeff)> from_euclidean_coeff;


    /**
     * \internal
     * \brief Wrap a given coefficient and return its wrapped, scalar value.
     * \param row The applicable row of the matrix.
     * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
     * returning its scalar value.
     * \return The scalar value of the wrapped coefficient corresponding to the provided
     * row and column (the column is an input into get_coeff).
     */
    std::function<Scalar(const std::size_t row, const GetCoeff& get_coeff)> wrap_get;


    /**
     * \internal
     * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
     * \param row The applicable row of the matrix.
     * \param s The value to set.
     * \param set_coeff A function that takes an index and a Scalar value, and uses that value to set
     * a coefficient in a matrix, without any wrapping.
     * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
     * returning its scalar value.
     */
    std::function<void(const std::size_t row, const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff)>
      wrap_set;


    /**
     * \brief Constructor taking a single \ref fixed_coefficients object.
     * \tparam C A \ref fixed_coefficients object.
     */
#ifdef __cpp_concepts
    template<fixed_coefficients C>
#else
    template<typename C, std::enable_if_t<fixed_coefficients<C>, int> = 0>
#endif
    DynamicCoefficients(C&&) :
      Dimensions<dynamic_size> {dimension_size_of_v<C>},
      runtime_dimension {dimension_size_of_v<C>},
      runtime_euclidean_dimension {euclidean_dimension_size_of_v<C>},
      id {typeid(reduced_fixed_index_descriptor_t<C>)},
      to_euclidean_coeff {[] (const std::size_t row, const GetCoeff& get_coeff) {
        return internal::to_euclidean_coeff<C>(row, get_coeff);
      }},
      from_euclidean_coeff {[] (const std::size_t row, const GetCoeff& get_coeff) {
        return internal::from_euclidean_coeff<C>(row, get_coeff);
      }},
      wrap_get {[] (const std::size_t row, const GetCoeff& get_coeff) {
        return internal::wrap_get<C>(row, get_coeff);
      }},
      wrap_set {[] (const std::size_t row, const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) {
        return internal::wrap_set<C>(row, s, set_coeff, get_coeff);
      }} {}


    /**
     * \brief Constructor taking multiple \ref fixed_coefficients objects.
     * \tparam Cs A list of \ref fixed_coefficients objects.
     */
#ifdef __cpp_concepts
    template<fixed_coefficients...Cs> requires (sizeof...(Cs) != 1)
#else
    template<typename...Cs, std::enable_if_t<(fixed_coefficients<Cs> and ...) and (sizeof...(Cs) != 1), int> = 0>
#endif
    DynamicCoefficients(Cs&&...) : DynamicCoefficients {Coefficients<Cs...> {}} {};


    DynamicCoefficients() : DynamicCoefficients {Coefficients<> {}} {};


#ifdef __cpp_impl_three_way_comparison
    /// \brief Three-way comparison with another DynamicCoefficients.
    auto operator<=>(const DynamicCoefficients& other) const { return id <=> other.id; }
#else
    /// \brief Compares for equivalence.
    bool operator==(const DynamicCoefficients& other) const { return id == other.id; }

    /// \brief Compares for non-equivalence.
    bool operator!=(const DynamicCoefficients& other) const { return id != other.id; }
#endif


  };


  /**
    * \internal
    * \brief The Euclidean size of DynamicCoefficients is not known at compile time.
    */
   template<typename Scalar>
   struct euclidean_dimension_size_of<DynamicCoefficients<Scalar>>
     : std::integral_constant<std::size_t, dynamic_size> {};


  /**
   * \internal
   * \brief The difference type for DynamicCoefficients is also DynamicCoefficients
   */
  template<typename Scalar>
  struct dimension_difference_of<DynamicCoefficients<Scalar>>
  {
    using type = DynamicCoefficients<Scalar>;
  };


} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICCOEFFICIENTS_HPP
