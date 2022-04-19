/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Dimensions class.
 */

#include <cstddef>

#ifndef OPENKALMAN_DIMENSIONS_HPP
#define OPENKALMAN_DIMENSIONS_HPP

namespace OpenKalman
{
  /**
   * \brief A structure representing the dimensions associated with of a particular index.
   * \details The dimension may or may not be known at compile time. If unknown at compile time, the size is set
   * at the time of construction and cannot be modified thereafter.
   * \tparam size The dimension (or <code>dynamic_size</code>, if not known at compile time)
   */
  template<std::size_t size = dynamic_size>
  struct Dimensions;


  /**
   * \overload
   * \brief The dimension or size associated with a given index known at compile time.
   */
  template<std::size_t size>
  struct Dimensions
  {
    /// The size of the dimension.
    static constexpr std::size_t value = size;

    /// \returns The size of the dimension as known at compile time
    constexpr std::size_t operator()() const
    {
      return size;
    }

    /// \returns The size of the dimension as known at compile time
    constexpr operator std::size_t() const
    {
      return size;
    }
  };


  /**
   * \overload
   * \brief The dimension or size associated with a given dynamic index known only at runtime.
   */
  template<>
  struct Dimensions<dynamic_size>
  {
  private:

    const std::size_t runtime_size;

  public:

    /// The size of the dimension (which in this case is <code>dynamic_size</code>).
    static constexpr std::size_t value = dynamic_size;

    Dimensions() = delete;

    /// Constructor, taking a dimension known at runtime.
    Dimensions(const std::size_t size) : runtime_size(size)
    {}

    /// \returns The size of the dimension as known at runtime
    const std::size_t operator()() const
    {
      return runtime_size;
    }

    /// \returns The size of the dimension as known at runtime
    operator std::size_t() const
    {
      return runtime_size;
    }
  };


#ifdef __cpp_concepts
  template<std::convertible_to<const std::size_t&> T>
#else
  template<typename T, std::enable_if_t<std::is_convertible<T, const std::size_t&>::value, int> = 0>
#endif
  Dimensions(T&&) -> Dimensions<dynamic_size>;


} // namespace OpenKalman


#endif //OPENKALMAN_DIMENSIONS_HPP
