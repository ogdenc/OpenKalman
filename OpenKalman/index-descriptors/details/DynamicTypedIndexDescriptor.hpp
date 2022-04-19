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
 * \brief Definition of the AbstractTypedIndexDescriptor and DynamicTypedIndexDescriptor classes.
 */

#ifndef OPENKALMAN_DYNAMICTTYPEDINDEXDESCRIPTORADAPTER_HPP
#define OPENKALMAN_DYNAMICTTYPEDINDEXDESCRIPTORADAPTER_HPP

#include <functional>

namespace OpenKalman
{
  /**
   * \internal
   * \brief The abstract class for dynamic typed index descriptors.
   */
  template<typename Scalar>
  struct AbstractTypedIndexDescriptor
  {
    /**
     * \internal
     * \brief A function taking a row index and returning a corresponding matrix element.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    /**
     * \internal
     * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
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
    virtual Scalar to_euclidean_coeff(const std::size_t row, const GetCoeff& get_coeff) const = 0;


    /**
     * \internal
     * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
     * \param row The applicable row of the transformed matrix.
     * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
     * returning its scalar value.
     * \return The scalar value of the typed coefficient corresponding to the provided row.
     */
    virtual Scalar from_euclidean_coeff(const std::size_t row, const GetCoeff& get_coeff) const = 0;


    /**
     * \internal
     * \brief Wrap a given coefficient and return its wrapped, scalar value.
     * \param row The applicable row of the matrix.
     * \return The scalar value of the wrapped coefficient corresponding to the provided
     * row and column (the column is an input into get_coeff).
     */
    virtual Scalar wrap_get(const std::size_t row, const GetCoeff& get_coeff) const = 0;


    /**
     * \internal
     * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
     * \param row The applicable row of the matrix.
     */
    virtual void wrap_set(const std::size_t row, const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) const = 0;

  };


  /**
   * \internal
   * \brief A dynamic adapter for a \ref fixed_coefficients.
   */
#ifdef __cpp_concepts
  template<fixed_coefficients FixedIndexDescriptor, typename Scalar>
#else
  template<typename FixedIndexDescriptor, typename Scalar>
#endif
  struct DynamicTypedIndexDescriptor : AbstractTypedIndexDescriptor<Scalar>
  {
#ifndef __cpp_concepts
    static_assert(fixed_coefficients<FixedIndexDescriptor>);
#endif

    /**
     * \internal
     * \brief A function taking a row index and returning a corresponding matrix element.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    /**
     * \internal
     * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
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
    Scalar to_euclidean_coeff(const std::size_t row, const GetCoeff& get_coeff) const
    {
      return FixedIndexDescriptor::template to_euclidean_array<Scalar, 0>[row](get_coeff);
    }


    /**
     * \internal
     * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
     * \param row The applicable row of the transformed matrix.
     * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
     * returning its scalar value.
     * \return The scalar value of the typed coefficient corresponding to the provided row.
     */
    Scalar from_euclidean_coeff(const std::size_t row, const GetCoeff& get_coeff) const
    {
      return FixedIndexDescriptor::template from_euclidean_array<Scalar, 0>[row](get_coeff);
    }


    /**
     * \internal
     * \brief Wrap a given coefficient and return its wrapped, scalar value.
     * \param row The applicable row of the matrix.
     * \return The scalar value of the wrapped coefficient corresponding to the provided
     * row and column (the column is an input into get_coeff).
     */
    Scalar wrap_get(const std::size_t row, const GetCoeff& get_coeff) const
    {
      return FixedIndexDescriptor::template wrap_array_get<Scalar, 0>[row](get_coeff);
    }


    /**
     * \internal
     * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
     * \param row The applicable row of the matrix.
     */
    void wrap_set(const std::size_t row, const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) const
    {
      FixedIndexDescriptor::template wrap_array_set<Scalar, 0>[row](s, set_coeff, get_coeff);
    }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICTTYPEDINDEXDESCRIPTORADAPTER_HPP
