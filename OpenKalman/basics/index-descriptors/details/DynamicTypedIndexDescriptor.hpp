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
 * \brief Definition of the AbstractDynamicTypedIndexDescriptor and DynamicTypedIndexDescriptor classes.
 */

#ifndef OPENKALMAN_DYNAMICTTYPEDINDEXDESCRIPTORADAPTER_HPP
#define OPENKALMAN_DYNAMICTTYPEDINDEXDESCRIPTORADAPTER_HPP

#include <tuple>

namespace OpenKalman
{
  /**
   * \internal
   * \brief A dynamic adapter for a \ref fixed_index_descriptor.
   * \tparam Scalar The scalar type associated with this index.
   */
#ifdef __cpp_concepts
  template<typename Scalar, fixed_index_descriptor FixedIndexDescriptor>
#else
  template<typename Scalar, typename FixedIndexDescriptor>
#endif
  struct DynamicTypedIndexDescriptor : AbstractDynamicTypedIndexDescriptor<Scalar>
  {
#ifndef __cpp_concepts
    static_assert(fixed_index_descriptor<FixedIndexDescriptor>);
#endif

    using Getter = std::function<Scalar(std::size_t)>;

    using Setter = std::function<void(Scalar, std::size_t)>;


    /**
     * \brief Maps an element to coordinates in Euclidean space.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
     * \param start The starting index within the index descriptor
     */
    virtual Scalar to_euclidean_element(const Getter& g, std::size_t euclidean_local_index, std::size_t start) const final
    {
      if constexpr (euclidean_index_descriptor<FixedIndexDescriptor>)
        return g(start + euclidean_local_index);
      else
        return FixedIndexDescriptor::template to_euclidean_element<Scalar>(g, euclidean_local_index, start);
    }


    /**
     * \brief Maps a coordinate in Euclidean space to an element.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting index within the Euclidean-transformed indices
     */
    virtual Scalar from_euclidean_element(const Getter& g, std::size_t local_index, std::size_t euclidean_start) const final
    {
      if constexpr (euclidean_index_descriptor<FixedIndexDescriptor>)
        return g(euclidean_start + local_index);
      else
        return FixedIndexDescriptor::template from_euclidean_element<Scalar>(g, local_index, euclidean_start);
    }


    /**
     * \brief Perform modular wrapping of an element.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
    virtual Scalar wrap_get_element(const Getter& g, std::size_t local_index, std::size_t start) const final
    {
      if constexpr (euclidean_index_descriptor<FixedIndexDescriptor>)
        return g(start + local_index);
      else
        return FixedIndexDescriptor::template wrap_get_element<Scalar>(g, local_index, start);
    }


    /**
     * \brief Set an element and then perform any necessary modular wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
    virtual void wrap_set_element(const Setter& s, const Getter& g, Scalar x, std::size_t local_index, std::size_t start) const final
    {
      if constexpr (euclidean_index_descriptor<FixedIndexDescriptor>)
        return s(x, start + local_index);
      else
        return FixedIndexDescriptor::template wrap_set_element<Scalar>(s, g, x, local_index, start);
    }


    /**
     * \brief Whether this index descriptor is untyped
     * \sa euclidean_index_descriptor
     */
    bool is_untyped() const final
    {
      return euclidean_index_descriptor<FixedIndexDescriptor>;
    }


    static DynamicTypedIndexDescriptor& get_instance()
    {
      static DynamicTypedIndexDescriptor instance;
      return instance;
    }


    DynamicTypedIndexDescriptor(const DynamicTypedIndexDescriptor&) = delete;

    DynamicTypedIndexDescriptor& operator=(const DynamicTypedIndexDescriptor&) = delete;

  private:

    DynamicTypedIndexDescriptor() = default;

  };


} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICTTYPEDINDEXDESCRIPTORADAPTER_HPP
