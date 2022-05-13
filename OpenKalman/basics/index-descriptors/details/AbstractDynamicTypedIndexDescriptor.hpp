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
 * \brief Definition of the AbstractDynamicTypedIndexDescriptor class.
 */

#ifndef OPENKALMAN_ABSTRACTDYNAMICTYPEDINDEXDESCRIPTOR_HPP
#define OPENKALMAN_ABSTRACTDYNAMICTYPEDINDEXDESCRIPTOR_HPP

#include <typeindex>

namespace OpenKalman
{
  /**
   * \internal
   * \brief The abstract class for dynamic typed index descriptors.
   * \tparam Scalar The scalar type associated with this index.
   */
  template<typename Scalar>
  struct AbstractDynamicTypedIndexDescriptor
  {
    using Getter = std::function<Scalar(std::size_t)>;

    using Setter = std::function<void(Scalar, std::size_t)>;


    virtual ~AbstractDynamicTypedIndexDescriptor() = default;


    /**
     * \brief Return a functor mapping elements of a matrix or other tensor to coordinates in Euclidean space.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a local index relative to the Euclidean-transformed coordinates (starting at 0); and
     * - the starting location of the index descriptor within any larger set of index type descriptors.
     */
    virtual std::function<Scalar(const Getter&, std::size_t euclidean_local_index, std::size_t start)>
    to_euclidean_element() const = 0;


    /**
     * \brief Return a functor mapping coordinates in Euclidean space to elements of a matrix or other tensor.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a local index relative to this index descriptor (starting at 0) accessing the element; and
     * - the starting location within any larger set of Euclidean-transformed index type descriptors.
     */
    virtual std::function<Scalar(const Getter&, std::size_t local_index, std::size_t euclidean_start)>
    from_euclidean_element() const = 0;


    /**
     * \brief Return a functor wrapping an angle.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a local index relative to this index descriptor (starting at 0) accessing the element; and
     * - the starting location of the index descriptor within any larger set of index type descriptors.
     */
    virtual std::function<Scalar(const Getter&, std::size_t local_index, std::size_t start)>
    wrap_get_element() const = 0;


    /**
     * \brief Return a functor setting an angle and then wrapping the angle.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>);
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a new value of type Scalar to set;
     * - a local index relative to this index descriptor (starting at 0) accessing the element; and
     * - the starting location of the index descriptor within any larger set of index type descriptors.
     */
    virtual std::function<void(const Setter&, const Getter&, Scalar x, std::size_t local_index, std::size_t start)>
    wrap_set_element() const = 0;


    /**
     * \brief Whether this index descriptor is untyped
     * \sa untyped_index_descriptor
     */
    virtual bool is_untyped() const = 0;


    /**
     * \brief Whether this index descriptor is composite
     * \sa composite_index_descriptor
     */
    virtual bool is_composite() const = 0;

  };


} // namespace OpenKalman


#endif //OPENKALMAN_ABSTRACTDYNAMICTYPEDINDEXDESCRIPTOR_HPP
