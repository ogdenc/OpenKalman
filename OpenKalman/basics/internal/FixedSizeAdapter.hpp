/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of FixedSizeAdapter.
 */

#ifndef OPENKALMAN_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_FIXEDSIZEADAPTER_HPP

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedMatrix, vector_space_descriptor...Vs> requires compatible_with_vector_space_descriptors<NestedMatrix, Vs...>
#else
  template<typename NestedMatrix, typename...Vs>
#endif
  struct FixedSizeAdapter : library_base_t<FixedSizeAdapter<NestedMatrix, Vs...>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(indexible<NestedMatrix>);
    static_assert((vector_space_descriptor<Vs> and ...));
    static_assert(compatible_with_vector_space_descriptors<NestedMatrix, Vs...>);
#endif


    /**
     * \brief Construct from compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg>
      requires (not std::derived_from<std::decay_t<Arg>, FixedSizeAdapter>)
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<FixedSizeAdapter, std::decay_t<Arg>>) and
      compatible_with_vector_space_descriptors<Arg, Vs...>, int> = 0>
#endif
    explicit FixedSizeAdapter(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \brief Construct from compatible indexible object based on a set of compatible \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg, vector_space_descriptor...Ids>
      requires (not fixed_size_adapter<Arg>) and (sizeof...(Ids) > 0)
#else
    template<typename Arg, typename...Ids, std::enable_if_t<(not fixed_size_adapter<Arg>) and
      compatible_with_vector_space_descriptors<Arg, Vs...> and (... and vector_space_descriptor<Ids>) and
      (sizeof...(Ids) > 0), int> = 0>
#endif
    FixedSizeAdapter(Arg&& arg, const Ids&...) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \overload
     * \brief Construct from another FixedSizeAdapter using a set of compatible \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, vector_space_descriptor...Ids> requires
      compatible_with_vector_space_descriptors<Arg, Vs...> and (sizeof...(Ids) > 0)
#else
    template<typename Arg, typename...Ids, std::enable_if_t<fixed_size_adapter<Arg> and (... and vector_space_descriptor<Ids>) and
      compatible_with_vector_space_descriptors<Arg, Vs...> and (sizeof...(Ids) > 0), int> = 0>
#endif
    FixedSizeAdapter(Arg&& arg, const Ids&...) noexcept : m_arg {std::forward<Arg>(arg).nested_object()} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_vector_space_descriptors<Arg, Vs...>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      m_arg = to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Get the nested object.
     */
    const auto& nested_object() const & noexcept { return m_arg; }

    /// \overload
    const auto&& nested_object() const && noexcept { return std::move(*this).m_arg; }


    /**
     * \brief Increment from another indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_vector_space_descriptors<Arg, Vs...>, int> = 0>
#endif
    auto& operator+=(Arg&& arg) noexcept
    {
      this->nested_object() += to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Decrement from another indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg>
#else
    template<typename Arg, std::enable_if_t<compatible_with_vector_space_descriptors<Arg, Vs...>, int> = 0>
#endif
    auto& operator-=(Arg&& arg) noexcept
    {
      this->nested_object() -= to_native_matrix<NestedMatrix>(std::forward<Arg>(arg));
      return *this;
    }

  private:

    NestedMatrix m_arg; //< The nested matrix.

  };


  // ----------------- //
  //  Deduction Guide  //
  // ----------------- //

#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor...Ids> requires (not fixed_size_adapter<Arg>) and (sizeof...(Ids) > 0)
#else
    template<typename Arg, typename...Ids, std::enable_if_t<indexible<Arg> and not fixed_size_adapter<Arg> and
      (... and vector_space_descriptor<Ids>) and (sizeof...(Ids) > 0), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Ids&...) -> FixedSizeAdapter<Arg, Ids...>;


#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, fixed_vector_space_descriptor...Ids> requires (sizeof...(Ids) > 0)
#else
    template<typename Arg, typename...Ids, std::enable_if_t<fixed_size_adapter<Arg> and
      (... and fixed_vector_space_descriptor<Ids>) and (sizeof...(Ids) > 0), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Ids&...) -> FixedSizeAdapter<nested_object_of_t<Arg>, Ids...>;


#ifdef __cpp_concepts
    template<one_dimensional<Likelihood::maybe> Arg>
#else
    template<typename Arg, std::enable_if_t<one_dimensional<Arg, Likelihood::maybe>, int> = 0>
#endif
    FixedSizeAdapter(Arg&&) -> FixedSizeAdapter<Arg>;


} // namespace OpenKalman::internal


#endif //OPENKALMAN_FIXEDSIZEADAPTER_HPP
