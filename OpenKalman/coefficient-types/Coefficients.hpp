/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COEFFICIENTS_H
#define OPENKALMAN_COEFFICIENTS_H

#include <array>
#include <functional>
#include <numeric>

namespace OpenKalman
{
  template<>
  struct Coefficients<>
  {
    static constexpr std::size_t size = 0;
    static constexpr std::size_t dimension = 0;
    static constexpr bool axes_only = true;
    using difference_type = Coefficients<>;

  private:
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);

    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);

  public:
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      to_Euclidean_array = {};

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      from_Euclidean_array = {};

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      wrap_array_get = {};

    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, size>
      wrap_array_set = {};

    template<typename ... Cnew>
    using Prepend = Coefficients<Cnew...>;

    template<typename ... Cnew>
    using Append = Coefficients<Cnew...>;

    template<std::size_t i>
    using Coefficient = Coefficients;

    template<std::size_t count>
    using Take = Coefficients;

    template<std::size_t count>
    using Discard = Coefficients;

    static_assert(internal::coefficient_class<Coefficients>);
  };


#ifdef __cpp_concepts
  template<coefficients C, coefficients ... Ctail>
#else
  template<typename C, typename ... Ctail>
#endif
  struct Coefficients<C, Ctail ...>
  {
#ifndef __cpp_concepts
    static_assert((coefficients<C> and ... and coefficients<Ctail>));
#endif
    static constexpr std::size_t size = C::size + Coefficients<Ctail...>::size; ///<Aggregate number of coefficients.

    /// Aggregate number of coefficients when converted to Euclidian.
    static constexpr std::size_t dimension = C::dimension + Coefficients<Ctail...>::dimension;

    /// Whether all the coefficients are of type Axis.
    static constexpr bool axes_only = C::axes_only and Coefficients<Ctail...>::axes_only;

    /**
     * \brief The type of the result when subtracting two Coefficients vectors.
     * \details Each coefficient is subtracted independently.
     */
    using difference_type = Concatenate<typename C::difference_type, typename Ctail::difference_type...>;

  private:
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

    template<typename Scalar>
    using GetCoeffFunction = Scalar(*)(const GetCoeff<Scalar>&);

    template<typename Scalar>
    using SetCoeffFunction = void(*)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&);

  public:
    /**
     * \internal
     * \brief An array of functions that convert the coefficients to coordinates in Euclidean space.
     * \details The functions in the array take the coefficients and convert them to
     * Cartesian coordinates in a Euclidean space, depending on the type of each coordinate.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first spherical coefficient that is being transformed.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, dimension>
      to_Euclidean_array = internal::join(C::template to_Euclidean_array<Scalar, i>,
        Coefficients<Ctail...>::template to_Euclidean_array<Scalar, i + C::size>);


    /**
     * \internal
     * \brief An array of functions that convert coordinates in Euclidean space into the typed coordinates.
     * \details The functions in the array take Cartesian coordinates, and convert them to the typed coordinates.
     * The array element is a function taking a ''get coefficient'' function and returning the typed coordinates.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns one of
     * the Cartesian coordinates.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first of the Cartesian coordinates being transformed back to their respective types.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      from_Euclidean_array = internal::join(C::template from_Euclidean_array<Scalar, i>,
        Coefficients<Ctail...>::template from_Euclidean_array<Scalar, i + C::dimension>);


    /**
     * \internal
     * \brief An array of functions that return a wrapped version of the coefficients.
     * \details Each function in the array takes a ''get coefficient'' function and returns wrapped coefficients.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns a coefficient.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first of the coefficients that are being wrapped.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const GetCoeffFunction<Scalar>, size>
      wrap_array_get = internal::join(C::template wrap_array_get<Scalar, i>,
        Coefficients<Ctail...>::template wrap_array_get<Scalar, i + C::size>);


    /**
     * \internal
     * \brief An array of functions that wraps and sets an existing matrix coefficient.
     * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
     * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
     * sets the coefficient at that index to a wrapped version of the scalar input.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam i The index of the first of the coefficients that are being wrapped.
     * This may be non-zero if this set of coefficients is part of a larger set of composite coordinates.
     */
    template<typename Scalar, std::size_t i>
    static constexpr std::array<const SetCoeffFunction<Scalar>, size>
      wrap_array_set = internal::join(C::template wrap_array_set<Scalar, i>,
        Coefficients<Ctail...>::template wrap_array_set<Scalar, i + C::size>);

    /**
     * \brief Prepend a set of new coefficients to the existing set.
     * \tparam Cnew The set of new coordinates to prepend.
     */
    template<typename ... Cnew>
    using Prepend = Coefficients<Cnew..., C, Ctail ...>;


    /**
     * \brief Append a set of new coordinates to the existing set.
     * \tparam Cnew The set of new coordinates to append.
     */
    template<typename ... Cnew>
    using Append = Coefficients<C, Ctail ..., Cnew ...>;


    /**
     * \brief Extract a particular coefficient from the set of coefficients.
     * \tparam i The index of the coefficient.
     */
    template<std::size_t i>
    using Coefficient = std::conditional_t<i == 0, C, typename Coefficients<Ctail...>::template Coefficient<i - 1>>;


    /**
     * \brief Take the first <code>count</code> coefficients.
     * \tparam count The number of coefficients to take.
     */
    template<std::size_t count>
    using Take = std::conditional_t<count == 0,
      Coefficients<>,
      typename Coefficients<Ctail...>::template Take<count - 1>::template Prepend<C>>;


    /**
     * \brief Discard all remaining coefficients after the first <code>count</code>.
     * \tparam count The index of the first coefficient to discard.
     */
    template<std::size_t count>
    using Discard = std::conditional_t<count == 0,
      Coefficients,
      typename Coefficients<Ctail...>::template Discard<count - 1>>;


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
   * \brief Alias for <code>Coefficients<Axis...></code>, where Axis is repeated <code>size</code> times.
   * \tparam size The number of Axes.
   */
  template<std::size_t size>
  using Axes = Replicate<Axis, size>;


}// namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_H
