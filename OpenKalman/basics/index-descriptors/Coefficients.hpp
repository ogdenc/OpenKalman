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
#include <tuple>

namespace OpenKalman
{
  /**
   * \brief A set of coefficient types.
   * \details This is the key to the wrapping functionality of OpenKalman. Each of the fixed_index_descriptor Cs... matches-up with
   * one or more of the rows or columns of a matrix. The number of coefficients per coefficient depends on the dimension
   * of the coefficient. For example, Axis, Distance, Angle, and Inclination are dimension 1, and each correspond to a
   * single coefficient. Polar is dimension 2 and corresponds to two coefficients (e.g., a distance and an angle).
   * Spherical is dimension 3 and corresponds to three coefficients.
   * Example: <code>Coefficients&lt;Axis, angle::Radians&gt;</code>
   * \sa Specializations: Coefficients<>, \ref CoefficientsCCs "Coefficients<C, Cs...>"
   * \tparam Cs Any types within the concept coefficients.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor...Cs>
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
  struct Coefficients<>
  {
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
  template<fixed_index_descriptor C, fixed_index_descriptor ... Cs>
#else
  template<typename C, typename ... Cs>
#endif
  struct Coefficients<C, Cs ...> : Dimensions<dimension_size_of_v<C> + dimension_size_of_v<Coefficients<Cs...>>>
  {
#ifndef __cpp_concepts
    static_assert((fixed_index_descriptor<C> and ... and fixed_index_descriptor<Cs>));
#endif

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

  private:
    /// Number of matrix rows corresponding to these coefficients.
    static constexpr std::size_t dimension = (dimension_size_of_v<C> + ... + dimension_size_of_v<Cs>);


    /// Number of matrix rows when these coefficients are converted to Euclidean space.
    static constexpr std::size_t
    euclidean_dimension = (euclidean_dimension_size_of_v<C> + ... + euclidean_dimension_size_of_v<Cs>);


    /*
     * \internal
     * \tparam euclidean Whether the relevant vector is in Euclidean space (true) or not (false)
     * \tparam i The row index
     * \tparam t The index (within dynamic_types) of C
     * \tparam local_index The local index for indices associated with each type (resets to 0 when t increments)
     * \tparam start The start location in the corresponding euclidean or non-euclidean vector
     * \return A tuple of tuples of {t, local_index, start}
     */
    template<bool euclidean, std::size_t i, std::size_t t, std::size_t local_index, std::size_t start, typename...Tups>
    static constexpr auto make_table(Tups&&...tups)
    {
      if constexpr (t >= 1 + sizeof...(Cs))
      {
        return std::array {std::forward<Tups>(tups)...};
      }
      else if constexpr (
        constexpr auto i_size = dimension_size_of_v<Coefficient<t>>, e_size = euclidean_dimension_size_of_v<Coefficient<t>>;
        local_index >= (euclidean ? e_size : i_size))
      {
        return make_table<euclidean, i, t + 1, 0, start + (euclidean ? i_size : e_size)>(std::forward<Tups>(tups)...);
      }
      else
      {
        return make_table<euclidean, i + 1, t, local_index + 1, start>(
          std::forward<Tups>(tups)..., std::tuple{t, local_index, start});
      }
    }


    static constexpr auto index_table = make_table<false, 0, 0, 0, 0>();


    static constexpr auto euclidean_index_table = make_table<true, 0, 0, 0, 0>();


    template<typename Scalar>
    using GRet = Scalar(*)(const std::function<Scalar(std::size_t)>&, std::size_t, std::size_t);


    template<typename Scalar>
    using SRet = void(*)(const std::function<void(Scalar, std::size_t)>&, const std::function<Scalar(std::size_t)>&,
      Scalar, std::size_t, std::size_t);


    template<typename Cx, typename Scalar> static constexpr GRet<Scalar> tfn()
    {
      return [](const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start) {
        return Cx::template to_euclidean_element<Scalar>(g, local_index, start); };
    }


    template<typename Scalar> static constexpr std::array<GRet<Scalar>, 1 + sizeof...(Cs)>
    to_euclidean_array { tfn<C, Scalar>(), tfn<Cs, Scalar>()... };


    template<typename Cx, typename Scalar> static constexpr GRet<Scalar> ffn()
    {
      return [](const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start) {
        return Cx::template from_euclidean_element<Scalar>(g, local_index, start); };
    }


    template<typename Scalar> static constexpr std::array<GRet<Scalar>, 1 + sizeof...(Cs)>
    from_euclidean_array { ffn<C, Scalar>(), ffn<Cs, Scalar>()... };


    template<typename Cx, typename Scalar> static constexpr GRet<Scalar> gsfn()
    {
      return [](const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start) {
        return Cx::template wrap_get_element<Scalar>(g, local_index, start); };
    }


    template<typename Scalar> static constexpr std::array<GRet<Scalar>, 1 + sizeof...(Cs)>
    wrap_get_array { gsfn<C, Scalar>(), gsfn<Cs, Scalar>()... };


    template<typename Cx, typename Scalar> static constexpr SRet<Scalar> wsfn()
    {
      return [](const std::function<void(Scalar, std::size_t)>& s, const std::function<Scalar(std::size_t)>& g,
          Scalar x, std::size_t local_index, std::size_t start) {
        return Cx::template wrap_set_element<Scalar>(s, g, x, local_index, start); };
    }


    template<typename Scalar> static constexpr std::array<SRet<Scalar>, 1 + sizeof...(Cs)>
    wrap_set_array { wsfn<C, Scalar>(), wsfn<Cs, Scalar>()... };

  public:

  /**
   * \brief Maps an element to coordinates in Euclidean space.
   * \tparam Scalar The scalar type (e.g., double).
   * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
   * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
   * \param start The starting index within the index descriptor
   */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    to_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t euclidean_local_index, std::size_t start)
    {
      auto [tp, comp_euclidean_local_index, comp_start] = euclidean_index_table[euclidean_local_index];
      return to_euclidean_array<Scalar>[tp](g, comp_euclidean_local_index, start + comp_start);
    }


  /**
   * \brief Maps a coordinate in Euclidean space to an element.
   * \tparam Scalar The scalar type (e.g., double).
   * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
   * \param local_index A local index relative to the original coordinates (starting at 0)
   * \param start The starting index within the Euclidean-transformed indices
   */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    from_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t euclidean_start)
    {
      auto [tp, comp_local_index, comp_euclidean_start] = index_table[local_index];
      return from_euclidean_array<Scalar>[tp](g, comp_local_index, euclidean_start + comp_euclidean_start);
    }


  /**
   * \brief Perform modular wrapping of an element.
   * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
   * \tparam Scalar The scalar type (e.g., double).
   * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
   * \param local_index A local index accessing the angle (in this case, it must be 0)
   * \param start The starting location of the angle within any larger set of index type descriptors
   */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    wrap_get_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start)
    {
      auto [tp, comp_local_index, comp_start] = index_table[local_index];
      return wrap_get_array<Scalar>[tp](g, comp_local_index, start + comp_start);
    }


  /**
   * \brief Set an element and then perform any necessary modular wrapping.
   * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
   * \tparam Scalar The scalar type (e.g., double).
   * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
   * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
   * \param local_index A local index accessing the angle (in this case, it must be 0)
   * \param start The starting location of the angle within any larger set of index type descriptors
   */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr void
    wrap_set_element(const std::function<void(Scalar, std::size_t)>& s, const std::function<Scalar(std::size_t)>& g,
      Scalar x, std::size_t local_index, std::size_t start)
    {
      auto [tp, comp_local_index, comp_start] = index_table[local_index];
      wrap_set_array<Scalar>[tp](s, g, x, comp_local_index, start + comp_start);
    }

  };


  /**
   * \brief The size is the sum of sizes of Cs.
   * \tparam Cs Component index descriptors
   */
  template<typename...Cs>
  struct dimension_size_of<Coefficients<Cs...>>
    : std::integral_constant<std::size_t, (dimension_size_of_v<Cs> + ... + 0)>
  {
    constexpr static std::size_t get(const Coefficients<Cs...>&) { return (dimension_size_of_v<Cs> + ... + 0); }
  };


  /**
   * \brief The size is the sum of Euclidean sizes of Cs.
   * \tparam Cs Component index descriptors
   */
  template<typename...Cs>
  struct euclidean_dimension_size_of<Coefficients<Cs...>>
    : std::integral_constant<std::size_t, (euclidean_dimension_size_of_v<Cs> + ... + 0)>
  {
    constexpr static std::size_t get(const Coefficients<Cs...>&) { return (euclidean_dimension_size_of_v<Cs> + ... + 0); }
  };


  /**
   * \brief The number of atomic components.
   * \tparam Cs Component index descriptors
   */
  template<typename...Cs>
  struct index_descriptor_components_of<Coefficients<Cs...>>
    : std::integral_constant<std::size_t, (0 + ... + index_descriptor_components_of<Cs>::value)>
  {
    constexpr static std::size_t get(const Coefficients<Cs...>&)
    {
      return (0 + ... + index_descriptor_components_of<Cs>::value);
    }
  };


  /**
   * \brief The concatenation of the difference types of Cs.
   * \tparam Cs Component index descriptors
   */
  template<typename...Cs>
  struct dimension_difference_of<Coefficients<Cs...>> { using type = Concatenate<dimension_difference_of_t<Cs>...>; };


  template<typename...Cs>
  struct is_untyped_index_descriptor<Coefficients<Cs...>>
    : std::bool_constant<(is_untyped_index_descriptor<Cs>::value and ...)> {};


}// namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_HPP
