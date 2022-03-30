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
  template<typename Scalar_>
  struct DynamicCoefficients
  {
    /// The scalar type of the coefficients
    using Scalar = Scalar_;

    /// The dimension at compile time.
    static constexpr std::size_t dimension = dynamic_size;

    /// The dimension when transformed to Euclidean space at compile time.
    static constexpr std::size_t euclidean_dimension = dynamic_size;

    /// The number of dimension at runtime.
    const std::size_t runtime_dimension;

    /// The number of coordinates in Euclidean space at runtime.
    const std::size_t runtime_euclidean_dimension;

    /// May consist of coefficients other than Axis.
    static constexpr bool axes_only = false;

    bool axes_only_at_runtime()
    {
      return true; // \todo implement this
    }

    /**
     * \brief The type of the result when subtracting two DynamicCoefficients values.
     * \details A difference between two dynamic coefficients is also dynamic.
     */
    using difference_type = DynamicCoefficients;

    /// The type index of the corresponding \ref fixed_coefficients.
    const std::type_index id;

  private:

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


#ifdef __cpp_concepts
    template<fixed_coefficients T>
#else
    template<typename T, typename = void>
#endif
    struct reduce_coeffs { using type = T; };


#ifdef __cpp_concepts
    template<fixed_coefficients T>
    struct reduce_coeffs<Coefficients<T>>
#else
    template<typename T>
    struct reduce_coeffs<Coefficients<T>, std::enable_if_t<fixed_coefficients<T>>>
#endif
    {
      using type = typename reduce_coeffs<T>::type;
    };


#ifdef __cpp_concepts
    template<template<typename...> typename T, fixed_coefficients...Cs> requires
      atomic_coefficient_group<T<Cs...>> or (sizeof...(Cs) != 1)
    struct reduce_coeffs<T<Cs...>>
#else
    template<template<typename...> typename T, typename...Cs>
    struct reduce_coeffs<T<Cs...>, std::enable_if_t<(atomic_coefficient_group<T<Cs...>> or (sizeof...(Cs) != 1))>>
#endif
    {
      using type = T<typename reduce_coeffs<Cs>::type...>;
    };


  public:

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
      runtime_dimension {C::dimension},
      runtime_euclidean_dimension {C::euclidean_dimension},
      id {typeid(typename reduce_coeffs<C>::type)},
      to_euclidean_coeff {[] (const std::size_t row, const GetCoeff& get_coeff) {
        return C::template to_euclidean_array<Scalar, 0>[row](get_coeff);
      }},
      from_euclidean_coeff {[] (const std::size_t row, const GetCoeff& get_coeff) {
          return C::template from_euclidean_array<Scalar, 0>[row](get_coeff);
      }},
      wrap_get {[] (const std::size_t row, const GetCoeff& get_coeff) {
        return C::template wrap_array_get<Scalar, 0>[row](get_coeff);
      }},
      wrap_set {[] (const std::size_t row, const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) {
        return C::template wrap_array_set<Scalar, 0>[row](s, set_coeff, get_coeff);
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


    /// \brief Compares for equivalence. \sa \ref equivalent_to
    bool operator==(const DynamicCoefficients& other) const { return id == other.id; }

#ifndef __cpp_impl_three_way_comparison
    /// Compares for non-equivalence. \sa \ref equivalent_to
    bool operator!=(const DynamicCoefficients& other) const { return id != other.id; }
#endif



#ifdef __cpp_concepts
    template<dynamic_coefficients Coeffs, typename F> requires
      requires(F& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
    template<typename Coeffs, typename F, typename>
#endif
    friend auto internal::to_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff);


#ifdef __cpp_concepts
    template<dynamic_coefficients Coeffs, typename F> requires
      requires(F& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
    template<typename Coeffs, typename F, typename>
#endif
    friend auto internal::from_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff);


#ifdef __cpp_concepts
    template<dynamic_coefficients Coeffs, typename F> requires
      requires(F& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
    template<typename Coeffs, typename F, typename>
#endif
    friend auto internal::wrap_get(Coeffs&& coeffs, const std::size_t row, const F& get_coeff);


#ifdef __cpp_concepts
    template<dynamic_coefficients Coeffs, typename FS, typename FG> requires
      requires(FS& f, std::size_t& i, typename Coeffs::Scalar& s) { f(i, s); } and
      requires(FG& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
    template<typename Coeffs, typename FS, typename FG, typename>
#endif
    friend void
    internal::wrap_set(Coeffs&& coeffs, const std::size_t row, const typename Coeffs::Scalar s,
                       const FS& set_coeff, const FG& get_coeff);


    static_assert(internal::coefficient_class<DynamicCoefficients>);

  };


} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICCOEFFICIENTS_HPP
