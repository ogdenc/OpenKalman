/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for StaticDescriptor class specializations and associated aliases.
 */

#ifndef OPENKALMAN_STATICDESCRIPTOR_HPP
#define OPENKALMAN_STATICDESCRIPTOR_HPP

#include <array>
#include <functional>
#include <numeric>

namespace OpenKalman::vector_space_descriptors
{
  /**
   * \brief A composite \ref static_vector_space_descriptor comprising a sequence of other fixed \ref vector_space_descriptor.
   * \details This is the key to the wrapping functionality of OpenKalman. Each of the static_vector_space_descriptor Cs... matches-up with
   * one or more of the rows or columns of a matrix. The number of coefficients per coefficient depends on the dimension
   * of the coefficient. For example, Axis, Distance, Angle, and Inclination are dimension 1, and each correspond to a
   * single coefficient. Polar is dimension 2 and corresponds to two coefficients (e.g., a distance and an angle).
   * Spherical is dimension 3 and corresponds to three coefficients.
   * Example: <code>StaticDescriptor&lt;Axis, angle::Radians&gt;</code>
   * \tparam Cs Any types within the concept coefficients.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs>
#else
  template<typename...Cs>
#endif
  struct StaticDescriptor
  {
#ifndef __cpp_concepts
    static_assert((static_vector_space_descriptor<Cs> and ...));
#endif

    /// Default constructor
    constexpr StaticDescriptor() = default;


    /// Conversion constructor
#ifdef __cpp_concepts
    template<maybe_equivalent_to<StaticDescriptor> D> requires (not std::same_as<std::decay_t<D>, StaticDescriptor>)
#else
    template<typename D, std::enable_if_t<
      maybe_equivalent_to<D, StaticDescriptor> and not std::is_same_v<std::decay_t<D>, StaticDescriptor>, int> = 0>
#endif
    explicit constexpr StaticDescriptor(D&& d)
    {
      if constexpr (dynamic_vector_space_descriptor<D>)
      {
        if (d != StaticDescriptor{}) throw std::invalid_argument{"Dynamic argument of 'StaticDescriptor' constructor is not an equivalent vector space descriptor."};
      }
    }


    /**
     * \brief Prepend a set of new \ref vector_space_descriptor to the existing set.
     * \tparam Cnew The set of new coordinates to prepend.
     */
    template<typename ... Cnew>
    using Prepend = StaticDescriptor<Cnew..., Cs ...>;


    /**
     * \brief Append a set of new coordinates to the existing set.
     * \tparam Cnew The set of new coordinates to append.
     */
    template<typename ... Cnew>
    using Append = StaticDescriptor<Cs ..., Cnew ...>;

  private:

    template<std::size_t count, typename...Ds>
    struct Take_impl { using type = StaticDescriptor<>; };


    template<std::size_t count, typename D, typename...Ds>
    struct Take_impl<count, D, Ds...> { using type = typename StaticDescriptor<Ds...>::template Take<count - 1>::template Prepend<D>; };


    template<typename D, typename...Ds>
    struct Take_impl<0, D, Ds...> { using type = StaticDescriptor<>; };

  public:

    /**
     * \brief Take the first <code>count</code> \ref vector_space_descriptor.
     * \tparam count The number of \ref vector_space_descriptor to take.
     */
#ifdef __cpp_concepts
    template<std::size_t count> requires (count <= sizeof...(Cs))
#else
    template<std::size_t count, std::enable_if_t<(count <= sizeof...(Cs)), int> = 0>
#endif
    using Take = typename Take_impl<count, Cs...>::type;


  private:

    template<std::size_t count, typename...Ds>
    struct Drop_impl { using type = StaticDescriptor<>; };


    template<std::size_t count, typename D, typename...Ds>
    struct Drop_impl<count, D, Ds...> { using type = typename StaticDescriptor<Ds...>::template Drop<count - 1>; };


    template<typename D, typename...Ds>
    struct Drop_impl<0, D, Ds...> { using type = StaticDescriptor<D, Ds...>; };

  public:

    /**
     * \brief Drop the first <code>count</code> \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template<std::size_t count> requires (count <= sizeof...(Cs))
#else
    template<std::size_t count, std::enable_if_t<(count <= sizeof...(Cs)), int> = 0>
#endif
    using Drop = typename Drop_impl<count, Cs...>::type;

  private:

    template<std::size_t i, typename...Ds>
    struct Select_impl;


    template<std::size_t i, typename D, typename...Ds>
    struct Select_impl<i, D, Ds...> { using type = typename StaticDescriptor<Ds...>::template Select<i - 1>; };


    template<typename D, typename...Ds>
    struct Select_impl<0, D, Ds...> { using type = D; };

  public:

    /**
     * \brief Extract a particular component from the set of fixed \ref vector_space_descriptor.
     * \tparam i The index of the \ref vector_space_descriptor component.
     */
#ifdef __cpp_concepts
    template<std::size_t i> requires (i < sizeof...(Cs))
#else
    template<std::size_t i, std::enable_if_t<(i < sizeof...(Cs)), int> = 0>
#endif
    using Select = typename Select_impl<i, Cs...>::type;

  }; // struct StaticDescriptor


  /**
   * \internal
   * \brief traits for StaticDescriptor.
   */
  template<typename...Cs>
  struct static_vector_space_descriptor_traits<StaticDescriptor<Cs...>>
  {
    static constexpr std::size_t size = (0 + ... + dimension_size_of_v<Cs>);


    static constexpr std::size_t euclidean_size = (0 + ... + euclidean_dimension_size_of_v<Cs>);


    static constexpr std::size_t component_count = (0 + ... + vector_space_component_count<Cs>::value);


    using difference_type = concatenate_static_vector_space_descriptor_t<dimension_difference_of_t<Cs>...>;


    static constexpr bool always_euclidean = (euclidean_vector_space_descriptor<Cs> and ...);

  private:

    // ------- Index tables ------- //

    /*
     * \internal
     * \tparam euclidean Whether the relevant vector is in Euclidean space (true) or not (false)
     * \tparam i The row index
     * \tparam t The component index within the list of descriptors
     * \tparam local_index The local index for indices associated with each of descriptors (resets to 0 when t increments)
     * \tparam start The start location in the corresponding euclidean or non-euclidean vector
     * \return An array of arrays of {t, local_index, start}
     */
    template<bool euclidean, std::size_t i, std::size_t t, std::size_t local_index, std::size_t start, typename...Arrs>
    static constexpr auto make_table(Arrs&&...arrs)
    {
      if constexpr (t < sizeof...(Cs))
      {
        using C = typename StaticDescriptor<Cs...>::template Select<t>;
        constexpr auto i_size = dimension_size_of_v<C>;
        constexpr auto e_size = euclidean_dimension_size_of_v<C>;
        if constexpr (local_index >= (euclidean ? e_size : i_size))
        {
          return make_table<euclidean, i, t + 1, 0, start + (euclidean ? i_size : e_size)>(std::forward<Arrs>(arrs)...);
        }
        else
        {
          return make_table<euclidean, i + 1, t, local_index + 1, start>(
            std::forward<Arrs>(arrs)..., std::array<std::size_t, 3> {t, local_index, start});
        }
      }
      else
      {
        return std::array<std::array<std::size_t, 3>, sizeof...(Arrs)> {std::forward<Arrs>(arrs)...};
      }
    }


    static constexpr auto index_table = make_table<false, 0, 0, 0, 0>();


    static constexpr auto euclidean_index_table = make_table<true, 0, 0, 0, 0>();


    // ------- Function tables ------- //


    template<typename Scalar>
    using GArr = std::array<Scalar(*)(const std::function<Scalar(std::size_t)>&, std::size_t, std::size_t),
      1 + sizeof...(Cs)>;


    template<typename Scalar>
    using SArr = std::array<void(*)(const std::function<void(const Scalar&, std::size_t)>&,
      const std::function<Scalar(std::size_t)>&, const Scalar&, std::size_t, std::size_t), 1 + sizeof...(Cs)>;


#ifdef __cpp_concepts
    template<typename Scalar> static constexpr GArr<Scalar>
    to_euclidean_array {static_vector_space_descriptor_traits<Cs>::to_euclidean_element... };

    template<typename Scalar> static constexpr GArr<Scalar>
    from_euclidean_array {static_vector_space_descriptor_traits<Cs>::from_euclidean_element... };

    template<typename Scalar> static constexpr GArr<Scalar>
    wrap_get_array {static_vector_space_descriptor_traits<Cs>::get_wrapped_component... };

    template<typename Scalar> static constexpr SArr<Scalar>
    wrap_set_array {static_vector_space_descriptor_traits<Cs>::set_wrapped_component... };
#else
  private:

    template<typename Scalar> using Getter = std::function<Scalar(std::size_t)>;
    template<typename Scalar> using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    template<typename Scalar> static constexpr GArr<Scalar>
    to_euclidean_array { static_vector_space_descriptor_traits<Cs>::template to_euclidean_element<Getter<Scalar>, 0>... };

    template<typename Scalar> static constexpr GArr<Scalar>
    from_euclidean_array { static_vector_space_descriptor_traits<Cs>::template from_euclidean_element<Getter<Scalar>, 0>... };

    template<typename Scalar> static constexpr GArr<Scalar>
    wrap_get_array { static_vector_space_descriptor_traits<Cs>::template get_wrapped_component<Getter<Scalar>, 0>... };

    template<typename Scalar> static constexpr SArr<Scalar>
    wrap_set_array { static_vector_space_descriptor_traits<Cs>::template set_wrapped_component<Setter<Scalar>, Getter<Scalar>, 0>... };
#endif

    static constexpr bool euclidean_type = (euclidean_vector_space_descriptor<Cs> and ...);

  public:

#ifdef __cpp_concepts
    static constexpr scalar_type auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto [tp, comp_euclidean_local_index, comp_start] = euclidean_index_table[euclidean_local_index];
      return to_euclidean_array<Scalar>[tp](g, comp_euclidean_local_index, start + comp_start);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[local_index];
      return from_euclidean_array<Scalar>[tp](g, comp_local_index, euclidean_start + comp_euclidean_start);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      using Scalar = decltype(g(std::declval<std::size_t>()));
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[local_index];
      return wrap_get_array<Scalar>[tp](g, comp_local_index, start + local_index - comp_local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); } and (euclidean_type or requires{ {x} -> scalar_type; })
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void set_wrapped_component(const S& s, const G& g,
      const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x, std::size_t local_index, std::size_t start)
#endif
    {
      using X = std::decay_t<decltype(x)>;
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[local_index];
      wrap_set_array<X>[tp](s, g, x, comp_local_index, start + local_index - comp_local_index);
    }

  };


}// namespace OpenKalman::vector_space_descriptors


#endif //OPENKALMAN_STATICDESCRIPTOR_HPP
